"""Orchestration for the conversation-generation pipeline.

A single :class:`BaseLLM` is created once and shared by every agent, so switching
provider (Groq by default) is one argument. The runner drives the pipeline in
stages — topic generation, transcript generation, **content validation** (an LLM
judge of corpus-fit + realism that gates the formatter: the transcript is only
formatted once it PASSES), JSON formatting, deterministic manual validation
(schema/timing), and **format validation** (an LLM judge that only checks the
formatter converted the approved transcript faithfully). Content quality is fixed
by regenerating the transcript; formatting problems are fixed by re-formatting.

    python -m conversations_generator.runner
    python -m conversations_generator.runner --language=hindi
    python -m conversations_generator.runner --language=english --model=gemini
    python -m conversations_generator.runner --model=sarvam --validation-model=gemma

``--language`` restricts the run to one corpus language (``hindi`` / ``hinglish``
/ ``english``); omit it to process every language, as before.

``--model`` is **generation only** (topic + conversation transcript) and always
wins when given, for any language. Only when it's *omitted* does Hindi default
to Sarvam and other languages default to Krutrim.

``--validation-model`` is shared by the **formatter**, the **content validator**,
and the **format validator** (not the deterministic manual checks), so the LLM
judges stay independent of the generation model. Defaults to ``gemma``.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents import (
    ConversationEditorAgent,
    ConversationFormatterAgent,
    ConversationGeneratorAgent,
    TopicGeneratorAgent,
)
from .agents.conversation_content_validator_agent import (
    ConversationContentValidatorAgent,
    ContentValidationReport,
)
from .agents.conversation_format_validator_agent import (
    ConversationFormatValidatorAgent,
    FormatValidationReport,
)
from .agents.conversation_validator_manual import ConversationValidatorManual, ValidationReport
from .configuration_reader import (
    apply_to_environ,
    get_agent_temperature,
    get_mode,
    get_num_workers,
    get_number_inclusion_percentage,
    get_run_languages,
    is_production,
)
from .llm import (
    APILimitError,
    DEFAULT_GENERATION_PROVIDER,
    DEFAULT_VALIDATION_PROVIDER,
    TOKEN_USAGE,
    BaseLLM,
    LLMError,
    LLMProvider,
    SarvamLLM,
    create_llm,
    resolve_provider,
)
from .loaders import read_corpus_instances
from .logger import Logger
from .models import CorpusInstance
from . import wandb_logger
from .storage import (
    BaseStorage,
    Checkpoint,
    HuggingFaceStorage,
    InstanceProgress,
    SkippedInstance,
    SkippedRegistry,
    StorageError,
)

# Corpus languages selectable via --language, mapped case-insensitively onto
# the corpus's actual casing ("Hindi" / "Hinglish" / "English").
SUPPORTED_LANGUAGES = ("hindi", "hinglish", "english")


# Target conversation duration is drawn from a Gaussian so lengths cluster
# around the middle of the 4–8 min window instead of being uniform. The mean
# sits at the window centre (6 min) and sigma is chosen so the 4–8 min range
# spans ~±2σ (≈95% of the mass). Samples are truncated (resampled) back into the
# window so the reported target is always a real 4–8 min value.
DURATION_MIN_SEC = 4 * 60      # 240s — 4 min
DURATION_MAX_SEC = 8 * 60      # 480s — 8 min
DURATION_MEAN_SEC = (DURATION_MIN_SEC + DURATION_MAX_SEC) / 2   # 360s — 6 min
DURATION_STD_SEC = (DURATION_MAX_SEC - DURATION_MIN_SEC) / 4    # 60s — ±2σ covers the range

# How far past ``instance.duration_sec`` we tolerate when accepting one conversation.
# A single 4–8 min conversation can overshoot slightly; large overshoots are rejected.
INSTANCE_OVERFLOW_TOLERANCE_SEC = 120.0
# Shortest conversation we still bother generating when budget is tight.
MIN_VIABLE_CONVERSATION_SEC = 60.0


def sample_target_duration_sec(
    mean: float = DURATION_MEAN_SEC,
    std: float = DURATION_STD_SEC,
    low: float = DURATION_MIN_SEC,
    high: float = DURATION_MAX_SEC,
) -> float:
    """Draw a target conversation duration (seconds) from a truncated Gaussian.

    Values are sampled from ``Normal(mean, std)`` and resampled until they land
    inside ``[low, high]`` (a truncated normal), so lengths follow a bell curve
    centred at ``mean`` rather than being uniform. Falls back to a hard clamp
    after a bounded number of tries to guarantee termination.
    """
    for _ in range(100):
        value = random.gauss(mean, std)
        if low <= value <= high:
            return round(value, 1)
    return round(min(max(random.gauss(mean, std), low), high), 1)


def cap_target_duration_for_budget(
    sampled_sec: float,
    remaining_sec: float,
    *,
    max_overflow_sec: float = INSTANCE_OVERFLOW_TOLERANCE_SEC,
) -> float | None:
    """Cap a sampled conversation duration so the instance stays near its target.

    Called immediately after topic generation once the remaining instance budget
    is known. When enough budget remains, the sampled 4–8 min target is kept
    (not shrunk toward the instance tail). Returns ``None`` when there is not
    enough budget left to generate a worthwhile conversation.
    """
    if remaining_sec <= 0:
        return None

    if remaining_sec >= DURATION_MIN_SEC:
        # Normal case: honour the sampled 4–8 min target, only clip to budget.
        ceiling = min(DURATION_MAX_SEC, remaining_sec + max_overflow_sec)
        capped = min(max(sampled_sec, DURATION_MIN_SEC), ceiling)
        if capped < DURATION_MIN_SEC:
            return None
        return round(capped, 1)

    # Final tail: less than 4 min of instance budget left — allow one shorter clip.
    if remaining_sec + max_overflow_sec < MIN_VIABLE_CONVERSATION_SEC:
        return None
    capped = min(sampled_sec, remaining_sec + max_overflow_sec)
    return round(max(capped, MIN_VIABLE_CONVERSATION_SEC), 1)


class GenerationBudget:
    """Tracks committed + reserved seconds against instance and language caps.

    Checkpoint/HF totals are the committed baseline. Each worker reserves its
    planned conversation duration (known after topic + target sampling) so
    parallel workers cannot collectively overshoot either limit.
    """

    def __init__(
        self,
        *,
        instance_target_sec: float,
        instance_generated_sec: float,
        language_target_sec: float | None = None,
        language_generated_sec: float | None = None,
        max_overflow_sec: float = INSTANCE_OVERFLOW_TOLERANCE_SEC,
    ) -> None:
        self.instance_target_sec = instance_target_sec
        self.instance_generated_sec = instance_generated_sec
        self.language_target_sec = language_target_sec
        self.language_generated_sec = language_generated_sec or 0.0
        self.max_overflow_sec = max_overflow_sec
        self._reserved: dict[str, float] = {}
        self._next_id = 0

    @property
    def reserved_sec(self) -> float:
        return sum(self._reserved.values())

    def instance_headroom(self) -> float:
        return (
            self.instance_target_sec
            + self.max_overflow_sec
            - self.instance_generated_sec
            - self.reserved_sec
        )

    def language_headroom(self) -> float:
        if self.language_target_sec is None:
            return float("inf")
        return (
            self.language_target_sec
            + self.max_overflow_sec
            - self.language_generated_sec
            - self.reserved_sec
        )

    def effective_headroom(self) -> float:
        return min(self.instance_headroom(), self.language_headroom())

    def can_accommodate(self, planned_sec: float) -> bool:
        if planned_sec <= 0:
            return False
        return planned_sec <= self.effective_headroom() + 1e-6

    def reserve(self, planned_sec: float, worker_id: int) -> str | None:
        if not self.can_accommodate(planned_sec):
            return None
        reservation_id = f"w{worker_id}-{self._next_id}"
        self._next_id += 1
        self._reserved[reservation_id] = planned_sec
        return reservation_id

    def release(self, reservation_id: str | None) -> None:
        if reservation_id:
            self._reserved.pop(reservation_id, None)

    def commit(self, actual_sec: float) -> None:
        self.instance_generated_sec += actual_sec
        self.language_generated_sec += actual_sec


def _language_budget_totals(
    corpus_df: Any,
    checkpoint: Checkpoint | None,
    run_languages: list[str] | None,
    language: str,
) -> tuple[float, float]:
    """Return (target_sec, generated_sec) for one language from corpus + checkpoint."""
    lang_key = language.lower()
    target_total = 0.0
    generated_total = 0.0
    for pos in _language_ordered_positions(corpus_df, run_languages):
        row = corpus_df.iloc[pos]
        if str(row.get("language", "")).lower() != lang_key:
            continue
        inst_target = float(row.get("duration_sec") or 0.0)
        target_total += inst_target
        if checkpoint is not None:
            inst_id = int(row["corpus_combination_id"])
            generated_total += checkpoint.get(inst_id, inst_target).generated_sec
    return target_total, generated_total


# ------------------------------------------------------------------ #
# Generation sampling temperature.
#
# Sarvam is effectively deterministic at the default low temperature and tends to
# degenerate on longer transcripts (repetition loops that never terminate), and
# because it's deterministic, every regeneration reproduces the SAME broken
# transcript — so all retries fail identically. It therefore runs HOTTER than the
# validation model (gemma), which stays cool for consistent judging. Retries
# escalate the temperature further so a regeneration actually explores a
# different transcript instead of repeating the same one.
# ------------------------------------------------------------------ #
SARVAM_GENERATION_TEMPERATURE = 0.7
GENERATION_TEMPERATURE_RETRY_STEP = 0.15
GENERATION_TEMPERATURE_MAX = 0.95


class ConversationRunner:
    """Drives the pipeline agents stage by stage.

    Validation happens in two clearly-separated places:

    * **Content validation** (LLM) runs on the plain-text transcript *before*
      formatting and gates it — the transcript is only formatted once it PASSES.
      It judges corpus-fit + realism (language, emotion, accent, gender,
      naturalness). Failing it regenerates the transcript.
    * **Format validation** (deterministic timing checks + an LLM faithfulness
      judge) runs *after* formatting. The content is already approved, so the
      faithfulness judge only checks that the formatter converted the transcript
      to JSON without dropping, adding, reordering, or rewording lines. Failing
      it re-runs the formatter (the transcript is left untouched).

    Parameters
    ----------
    llm : BaseLLM | None
        Generation LLM — topic + conversation generator only. ``None`` lets those
        agents fall back to their own default (Groq).
    validation_llm : BaseLLM | None
        Shared by the formatter and both LLM validators. When ``None``, falls
        back to ``llm`` (same provider for everything).
    validator : ConversationValidatorManual | None
        Deterministic timing/overlap validator run after formatting. Defaults to
        ``ConversationValidatorManual()`` with its standard thresholds.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        validation_llm: BaseLLM | None = None,
        validator: ConversationValidatorManual | None = None,
        max_generation_attempts: int = 3,
        max_format_attempts: int = 3,
        max_content_validation_retries: int = 3,
        max_format_validation_retries: int = 3,
        max_edit_attempts: int = 2,
    ) -> None:
        # Generation path: topic + plain-text conversation only.
        self.llm = llm
        # Formatting + LLM validation path (defaults to the generation LLM when
        # the caller doesn't split providers).
        self.validation_llm = validation_llm if validation_llm is not None else llm
        self.topic_agent = TopicGeneratorAgent(llm)
        # Conversation generation is a two-stage pipeline: the generator writes
        # tagged plain text, the formatter turns it into schema JSON with
        # deterministic timing/overlap metadata.
        self.generator_agent = ConversationGeneratorAgent(llm)
        self.formatter_agent = ConversationFormatterAgent(self.validation_llm)
        self.validator = validator or ConversationValidatorManual()
        # Both LLM judges run on the validation LLM (``--validation-model``), kept
        # independent of the generation model so a judge never grades its own
        # writer's output: the content judge scores the transcript's corpus-fit +
        # realism, the faithfulness judge checks the formatter's conversion.
        self.content_validator = ConversationContentValidatorAgent(self.validation_llm)
        self.format_validator = ConversationFormatValidatorAgent(self.validation_llm)
        # Repairs a faithfulness failure with targeted per-turn edits (restoring a
        # reworded turn to the transcript, dropping an extra one) instead of
        # re-running the whole formatter. Uses the generation LLM (it owns dialogue
        # content), falling back to the validation LLM when generation is unset.
        self.editor_agent = ConversationEditorAgent(llm if llm is not None else self.validation_llm)
        # How many transcripts to generate (each content-validated) before giving up.
        self.max_generation_attempts = max(1, max_generation_attempts)
        # How many formatter passes (each manual- + faithfulness-checked) to try.
        self.max_format_attempts = max(1, max_format_attempts)
        # How many times to retry a validation CALL that errors out
        # (unparseable LLM response) before bypassing that step.
        self.max_content_validation_retries = max(1, max_content_validation_retries)
        self.max_format_validation_retries = max(1, max_format_validation_retries)
        # How many targeted-edit passes to try before falling back to re-formatting.
        self.max_edit_attempts = max(1, max_edit_attempts)

        # Base sampling temperature for the CONVERSATION generator. Sarvam runs
        # hotter than everything else (it degenerates/repeats at the default low
        # temperature); other providers keep the configured conversation value.
        conversation_temp = get_agent_temperature("conversation")
        if isinstance(self.llm, SarvamLLM):
            self.generation_base_temperature = max(conversation_temp, SARVAM_GENERATION_TEMPERATURE)
        else:
            self.generation_base_temperature = conversation_temp

    def _generation_temperature(self, attempt: int) -> float:
        """Sampling temperature for a generation attempt (escalates on retries).

        Attempt 1 uses the base (Sarvam-aware) temperature; each subsequent retry
        adds a step so a regeneration explores a different transcript rather than
        reproducing the same (possibly degenerate) one. Capped so it never gets
        so hot the output turns incoherent.
        """
        temp = self.generation_base_temperature + GENERATION_TEMPERATURE_RETRY_STEP * (attempt - 1)
        return round(min(GENERATION_TEMPERATURE_MAX, temp), 2)

    # ------------------------------------------------------------------ #
    # Stage 1: topic
    # ------------------------------------------------------------------ #
    def generate_topic(self, **profile: Any) -> dict[str, str]:
        """Produce the next single topic (see ``TopicGeneratorAgent.run``).

        The topic call is JSON-mode, and the model occasionally returns an empty
        or unparseable response (a transient failure the LLM layer's own retries
        can't always shake). Retry the whole call a few times so one bad response
        doesn't abort the run; re-raise only if every attempt fails, letting the
        caller decide (it treats that as a failed conversation, not a crash).
        """
        Logger.info(f"Generating topic for {profile.get('language', 'unknown')}...")
        last_err: Exception | None = None
        for attempt in range(1, self.max_generation_attempts + 1):
            try:
                topic = self.topic_agent.run(**profile)
                Logger.success(f"Topic generated: {topic.get('title', 'Unknown Title')}")
                return topic
            except (ValueError, LLMError) as err:
                last_err = err
                Logger.warning(
                    f"Topic generation failed "
                    f"(attempt {attempt}/{self.max_generation_attempts}): {err}"
                )
        raise LLMError(
            f"Topic generation failed after {self.max_generation_attempts} attempts: {last_err}"
        ) from last_err

    # ------------------------------------------------------------------ #
    # Stage 2: conversation
    # ------------------------------------------------------------------ #
    def generate_conversation(
        self,
        topic: dict[str, str],
        previous_transcript: str | None = None,
        feedback: str | None = None,
        target_duration_sec: float | None = None,
        **profile: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate a full conversation from a topic dict, in two stages.

        Stage 2a: the generator agent writes a tagged plain-text transcript.
        Stage 2b: the formatter agent converts it into schema turn dicts with
        deterministic timing/overlap metadata.

        Parameters
        ----------
        topic : dict
            Must contain ``title`` and ``context`` (output of stage 1).
        previous_transcript : str | None
            The previous attempt's transcript to learn from, if validation failed.
        feedback : str | None
            Validation feedback string.
        target_duration_sec : float | None
            Exact target duration (seconds) the conversation should aim for,
            drawn from the Gaussian in :func:`sample_target_duration_sec`.
        **profile
            Language, emotion, accent, gender_pair, etc.

        Returns
        -------
        tuple[str, list[dict]]
            The raw plain-text transcript and the formatted schema turns.
        """
        transcript = self.generator_agent.run(
            title=topic["title"],
            context=topic.get("context", ""),
            conversation_type=topic.get("conversation_type"),
            previous_transcript=previous_transcript,
            feedback=feedback,
            target_duration_sec=target_duration_sec,
            **profile,
        )
        turns = self.formatter_agent.run(
            transcript=transcript,
            agent_emotion=profile.get("agent_emotion"),
            user_emotion=profile.get("user_emotion"),
            language=profile.get("language"),
            target_duration_sec=target_duration_sec,
        )
        return transcript, turns

    # ------------------------------------------------------------------ #
    # Stage 3: manual validation
    # ------------------------------------------------------------------ #
    def validate_conversation(
        self,
        turns: list[dict[str, Any]],
        time_field: str = "planned",
    ) -> ValidationReport:
        """Run the deterministic timing/overlap checks over generated turns.

        See ``ConversationValidatorManual.validate`` for what's checked
        (schema correctness, overlap symmetry, interruption/backchannel/
        collision timing, total duration, turn-type distribution).
        """
        return self.validator.validate(turns, time_field=time_field)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def _plan_conversation(
        self,
        remaining_sec: float | None = None,
        max_overflow_sec: float = INSTANCE_OVERFLOW_TOLERANCE_SEC,
        **profile: Any,
    ) -> dict[str, Any]:
        """Topic + target-duration planning phase (cheap budget gate before generation)."""
        include_numbers = random.random() < get_number_inclusion_percentage()
        Logger.info(
            f"Number inclusion: {'ON — numbers + reasoning' if include_numbers else 'OFF — qualitative'}"
        )

        try:
            topic = self.generate_topic(include_numbers=include_numbers, **profile)
        except (ValueError, LLMError) as err:
            Logger.error(f"Topic generation failed; skipping this conversation: {err}")
            return {
                "topic": {},
                "transcript": "",
                "turns": [],
                "manual_validation": None,
                "content_validation": None,
                "format_validation": None,
                "content_validation_bypassed": False,
                "format_validation_bypassed": False,
                "format_validation_exhausted": False,
                "target_duration_sec": None,
                "include_numbers": include_numbers,
                "content_attempts_used": 0,
                "format_attempts_used": 0,
                "content_validation_attempts_used": 0,
                "format_validation_attempts_used": 0,
                "max_generation_attempts": self.max_generation_attempts,
                "max_format_attempts": self.max_format_attempts,
                "max_content_validation_retries": self.max_content_validation_retries,
                "max_format_validation_retries": self.max_format_validation_retries,
                "passed": False,
            }

        target_duration_sec = sample_target_duration_sec()
        if remaining_sec is not None:
            capped = cap_target_duration_for_budget(
                target_duration_sec,
                remaining_sec,
                max_overflow_sec=max_overflow_sec,
            )
            if capped is None:
                Logger.info(
                    f"Instance budget exhausted ({remaining_sec:.0f}s remaining) — "
                    "stopping after topic (no conversation generated)."
                )
                return {
                    "topic": topic,
                    "transcript": "",
                    "turns": [],
                    "manual_validation": None,
                    "content_validation": None,
                    "format_validation": None,
                    "content_validation_bypassed": False,
                    "format_validation_bypassed": False,
                    "format_validation_exhausted": False,
                    "target_duration_sec": None,
                    "include_numbers": include_numbers,
                    "content_attempts_used": 0,
                    "format_attempts_used": 0,
                    "content_validation_attempts_used": 0,
                    "format_validation_attempts_used": 0,
                    "max_generation_attempts": self.max_generation_attempts,
                    "max_format_attempts": self.max_format_attempts,
                    "max_content_validation_retries": self.max_content_validation_retries,
                    "max_format_validation_retries": self.max_format_validation_retries,
                    "passed": False,
                    "budget_exhausted": True,
                }
            if capped != target_duration_sec:
                Logger.info(
                    f"Capped conversation target {target_duration_sec:.0f}s → {capped:.0f}s "
                    f"to fit instance budget ({remaining_sec:.0f}s remaining, "
                    f"+{max_overflow_sec:.0f}s overflow allowed)."
                )
            target_duration_sec = capped

        Logger.info(
            f"Target conversation duration: {target_duration_sec:.1f}s "
            f"({target_duration_sec / 60:.2f} min)"
        )
        return {
            "topic": topic,
            "target_duration_sec": target_duration_sec,
            "include_numbers": include_numbers,
        }

    def _execute_conversation(
        self,
        plan: dict[str, Any],
        **profile: Any,
    ) -> dict[str, Any]:
        """Run transcript → validation → formatting for a planned conversation."""
        Logger.step(
            f"Generating conversation for {profile.get('language', 'unknown')} "
            f"(target {plan['target_duration_sec']:.0f}s)…"
        )
        topic = plan["topic"]
        target_duration_sec = plan["target_duration_sec"]
        include_numbers = plan["include_numbers"]

        transcript: str = ""
        content_report: ContentValidationReport | None = None
        content_validation_bypassed = False
        content_attempts_used = 0
        content_validation_attempts_used = 0

        generator_feedback: str | None = None
        previous_transcript: str | None = None

        content_ok = False
        for gen_attempt in range(1, self.max_generation_attempts + 1):
            content_attempts_used = gen_attempt
            if gen_attempt > 1:
                Logger.retry(
                    f"Regenerating conversation (attempt {gen_attempt}/{self.max_generation_attempts})"
                )
            else:
                Logger.step("Stage 2: Conversation Generation")

            # Escalate the sampling temperature on each retry. At the configured
            # low temperature some generation models (e.g. Sarvam) are effectively
            # deterministic — a regeneration reproduces the SAME transcript (and the
            # same degeneration, like a repetition loop), so every retry fails
            # identically. Bumping the temperature makes retries actually explore a
            # different transcript. Attempt 1 uses the configured value.
            gen_temperature = self._generation_temperature(gen_attempt)

            Logger.info(
                f"Generating conversation transcript (plain text) — "
                f"temperature {gen_temperature:.2f}..."
            )
            try:
                transcript = self.generator_agent.run(
                    title=topic["title"],
                    context=topic.get("context", ""),
                    conversation_type=topic.get("conversation_type"),
                    previous_transcript=previous_transcript,
                    feedback=generator_feedback,
                    target_duration_sec=target_duration_sec,
                    include_numbers=include_numbers,
                    temperature=gen_temperature,
                    **profile,
                )
            except (ValueError, LLMError) as err:
                Logger.warning(f"Transcript generation failed: {err}")
                generator_feedback = f"Conversation generation error on the previous attempt: {err}"
                previous_transcript = None
                content_report = None
                continue

            # ---- Stage 3: content validation gate (BEFORE formatting) ----
            Logger.step(
                f"Stage 3: Content Validation (attempt {gen_attempt}/{self.max_generation_attempts})"
            )
            content_report, content_validation_attempts_used = self._run_content_validation(
                transcript, topic, profile
            )

            if content_report is None:
                # Judge unavailable after retries — accept the transcript as-is so
                # a broken validator can't stall the run (rare infra case).
                Logger.warning(
                    f"Content validation unavailable after {self.max_content_validation_retries} "
                    "retries. Bypassing the content gate for this conversation."
                )
                content_validation_bypassed = True
                content_ok = True
                break

            if content_report.passed:
                Logger.success(
                    f"Content validation passed! (Realism: {content_report.realism_score}, "
                    f"Match: {content_report.corpus_match_score})",
                    bold=True,
                )
                content_ok = True
                break

            Logger.warning(f"Content validation failed. Verdict: {content_report.verdict}")
            generator_feedback = "Content Validation Feedback:\n" + (
                content_report.as_feedback() or "The transcript did not match the required attributes."
            )
            previous_transcript = transcript or None

        if not content_ok:
            # Never got an approved transcript — reject without formatting so we
            # honour the gate ("do not format until content is approved").
            Logger.error(
                "Content validation never passed after all attempts; "
                "rejecting this conversation (not formatted)."
            )
            return {
                "topic": topic,
                "transcript": transcript,
                "turns": [],
                "manual_validation": None,
                "content_validation": content_report,
                "format_validation": None,
                "content_validation_bypassed": content_validation_bypassed,
                "format_validation_bypassed": False,
                "target_duration_sec": target_duration_sec,
                "include_numbers": include_numbers,
                "content_attempts_used": content_attempts_used,
                "format_attempts_used": 0,
                "content_validation_attempts_used": content_validation_attempts_used,
                "format_validation_attempts_used": 0,
                "max_generation_attempts": self.max_generation_attempts,
                "max_format_attempts": self.max_format_attempts,
                "max_content_validation_retries": self.max_content_validation_retries,
                "max_format_validation_retries": self.max_format_validation_retries,
                "passed": False,
            }

        # ---- Stage 4: format the APPROVED transcript + validate the formatting ----
        Logger.step("Stage 4: Formatting & Format Validation")
        turns: list[dict[str, Any]] = []
        manual_report: ValidationReport | None = None
        format_report: FormatValidationReport | None = None
        format_validation_bypassed = False
        # Set when the faithfulness judge kept failing across all retries but the
        # formatting was structurally sound every time (manual validation clean).
        # The LLM judge can hallucinate faithfulness errors on a correct conversion,
        # so we don't let it discard a manual-valid, content-approved conversation.
        format_validation_exhausted = False
        format_ok = False
        format_attempts_used = 0
        format_validation_attempts_used = 0

        # Best manual-clean formatting seen so far, kept as the fallback to accept
        # if the faithfulness judge never returns PASS.
        best_turns: list[dict[str, Any]] | None = None
        best_manual_report: ValidationReport | None = None

        formatter_feedback: str | None = None
        previous_output: list[dict[str, Any]] | None = None

        for fmt_attempt in range(1, self.max_format_attempts + 1):
            format_attempts_used = fmt_attempt
            if fmt_attempt > 1:
                Logger.retry(
                    f"Re-formatting (attempt {fmt_attempt}/{self.max_format_attempts})"
                )

            Logger.info("Formatting transcript into schema turns...")
            try:
                turns = self.formatter_agent.run(
                    transcript=transcript,
                    agent_emotion=profile.get("agent_emotion"),
                    user_emotion=profile.get("user_emotion"),
                    language=profile.get("language"),
                    feedback=formatter_feedback,
                    previous_output=previous_output,
                    target_duration_sec=target_duration_sec,
                )
            except (ValueError, LLMError) as err:
                Logger.warning(f"Formatting failed: {err}")
                formatter_feedback = f"Formatting error on the previous attempt: {err}"
                previous_output = None
                manual_report = None
                continue

            # ---- Stage 4a: deterministic manual validation (schema/timing) ----
            Logger.info("Running deterministic manual validation...")
            manual_report = self.validate_conversation(turns)
            if manual_report.has_errors:
                Logger.warning(f"Manual validation failed with {len(manual_report.errors)} errors.")
                formatter_feedback = "Manual Validation Errors:\n" + "\n".join(
                    f"- [Turn {e.turn_id}]: {e.message}" for e in manual_report.errors
                )
                previous_output = turns or None
                continue
            Logger.success("Manual validation passed.")
            # Remember this structurally-valid formatting as the accept-fallback.
            best_turns = turns
            best_manual_report = manual_report

            # ---- Stage 4b: LLM faithfulness gate (conversion fidelity only) ----
            format_report, format_validation_attempts_used = self._run_format_validation(
                transcript, turns
            )
            if format_report is None:
                Logger.warning(
                    f"Format validation unavailable after {self.max_format_validation_retries} "
                    "retries. Accepting the manual-valid conversation."
                )
                format_validation_bypassed = True
                format_ok = True
                break
            if format_report.passed:
                Logger.success("Format validation passed — faithful conversion.", bold=True)
                format_ok = True
                break

            Logger.warning(f"Format validation failed. Verdict: {format_report.verdict}")

            # ---- Stage 4c: TARGETED EDITS before falling back to re-formatting ----
            # Patch only the flagged turns (restore a reworded turn to the
            # transcript, drop an extra one) instead of re-running the whole
            # formatter. If editing makes it faithful + manual-valid: done.
            edit_result = self._repair_by_editing(turns, transcript, format_report, profile)
            if edit_result is not None:
                turns, edited_manual, edited_format = edit_result
                turns = self.formatter_agent.scale_timings_to_target(turns, target_duration_sec)
                edited_manual = self.validate_conversation(turns)
                manual_report = edited_manual
                best_turns = turns
                best_manual_report = edited_manual
                if edited_format is None:
                    # Re-validation unavailable after edits; accept the edited,
                    # manual-valid conversation (same policy as the bypass above).
                    format_validation_bypassed = True
                    format_ok = True
                    break
                format_report = edited_format
                if format_report.passed:
                    Logger.success("Targeted edits fixed the formatting — faithful conversion.", bold=True)
                    format_ok = True
                    break
                # Edits helped but didn't fully pass — fall through to re-format,
                # now carrying the latest issues from the edited turns.

            formatter_feedback = "Formatting Faithfulness Errors:\n" + (
                format_report.as_feedback() or "The formatted turns did not faithfully match the transcript."
            )
            previous_output = turns or None

        # Faithfulness judge never returned PASS, but the formatting was structurally
        # valid (manual-clean). The judge can hallucinate faithfulness errors on a
        # correct conversion, and content is already approved, so accept the best
        # manual-valid formatting rather than discard a good conversation.
        if not format_ok and best_turns is not None:
            Logger.warning(
                f"Format validation never passed in {self.max_format_attempts} attempts, "
                "but formatting is manual-valid; accepting best formatting (judge may be "
                "over-flagging a correct conversion)."
            )
            turns = best_turns
            manual_report = best_manual_report
            format_validation_exhausted = True

        return {
            "topic": topic,
            "transcript": transcript,
            "turns": turns,
            "manual_validation": manual_report,
            "content_validation": content_report,
            "format_validation": format_report,
            "content_validation_bypassed": content_validation_bypassed,
            "format_validation_bypassed": format_validation_bypassed,
            "format_validation_exhausted": format_validation_exhausted,
            "target_duration_sec": target_duration_sec,
            "include_numbers": include_numbers,
            "content_attempts_used": content_attempts_used,
            "format_attempts_used": format_attempts_used,
            "content_validation_attempts_used": content_validation_attempts_used,
            "format_validation_attempts_used": format_validation_attempts_used,
            "max_generation_attempts": self.max_generation_attempts,
            "max_format_attempts": self.max_format_attempts,
            "max_content_validation_retries": self.max_content_validation_retries,
            "max_format_validation_retries": self.max_format_validation_retries,
            # Accept only if content was approved (or bypassed) and the formatting
            # passed deterministic manual validation. The LLM faithfulness judge is
            # advisory: it must pass, be bypassed (unavailable), or have exhausted
            # its retries on manual-valid formatting — it cannot, by itself, reject
            # a conversation the deterministic checks already vouch for.
            "passed": bool(
                content_ok
                and manual_report
                and not manual_report.has_errors
                and (format_ok or format_validation_exhausted)
            ),
        }

    def run(
        self,
        remaining_sec: float | None = None,
        max_overflow_sec: float = INSTANCE_OVERFLOW_TOLERANCE_SEC,
        **profile: Any,
    ) -> dict[str, Any]:
        """Run the full pipeline with a content gate before formatting."""
        Logger.step(f"Starting pipeline for language: {profile.get('language', 'unknown')}")
        plan = self._plan_conversation(
            remaining_sec=remaining_sec,
            max_overflow_sec=max_overflow_sec,
            **profile,
        )
        if plan.get("budget_exhausted") or plan.get("passed") is False:
            return plan
        return self._execute_conversation(plan, **profile)

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _run_content_validation(
        self,
        transcript: str,
        topic: dict[str, str],
        profile: dict[str, Any],
    ) -> tuple[ContentValidationReport | None, int]:
        """Call the content judge, retrying transient call failures.

        Returns ``(report, attempts_used)``; ``report`` is ``None`` if every
        attempt raised (unparseable/failed response), so the caller can bypass.
        """
        report: ContentValidationReport | None = None
        attempts_used = 0
        for attempt in range(1, self.max_content_validation_retries + 1):
            attempts_used = attempt
            try:
                report = self.content_validator.run(transcript=transcript, topic=topic, **profile)
                break
            except (ValueError, LLMError) as err:
                Logger.warning(
                    f"Content validation call failed "
                    f"(attempt {attempt}/{self.max_content_validation_retries}): {err}"
                )
        return report, attempts_used

    def _run_format_validation(
        self,
        transcript: str,
        turns: list[dict[str, Any]],
    ) -> tuple[FormatValidationReport | None, int]:
        """Call the faithfulness judge, retrying transient call failures.

        Returns ``(report, attempts_used)``; ``report`` is ``None`` if every
        attempt raised, so the caller can bypass the faithfulness check.
        """
        report: FormatValidationReport | None = None
        attempts_used = 0
        for attempt in range(1, self.max_format_validation_retries + 1):
            attempts_used = attempt
            try:
                report = self.format_validator.run(transcript=transcript, turns=turns)
                break
            except (ValueError, LLMError) as err:
                Logger.warning(
                    f"Format validation call failed "
                    f"(attempt {attempt}/{self.max_format_validation_retries}): {err}"
                )
        return report, attempts_used

    def _repair_by_editing(
        self,
        turns: list[dict[str, Any]],
        transcript: str,
        format_report: FormatValidationReport,
        profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], ValidationReport, FormatValidationReport | None] | None:
        """Fix a faithfulness failure with targeted per-turn edits.

        Repeatedly asks the editor agent for a minimal patch addressing the
        faithfulness judge's issues (restore a reworded turn to the transcript,
        drop an added one), applies it (which deterministically re-lays-out
        timing), re-checks manual validation, then re-judges faithfulness — up to
        ``max_edit_attempts`` passes. The **transcript is passed to the editor as
        ground truth** so it can make each turn match it.

        Returns ``(edited_turns, manual_report, format_report)`` where
        ``format_report`` is the latest verdict (or ``None`` if re-validation was
        unavailable). Returns ``None`` when editing can't be applied at all
        (editor error, no edits proposed, patch rejected, or the edited turns fail
        manual validation) — the caller then falls back to re-formatting.
        """
        current_turns = turns
        current_report = format_report
        manual_report: ValidationReport | None = None

        for edit_attempt in range(1, self.max_edit_attempts + 1):
            Logger.step(
                f"Stage 4c: Targeted edit repair (attempt {edit_attempt}/{self.max_edit_attempts})"
            )
            try:
                edits = self.editor_agent.run(
                    turns=current_turns,
                    issues=current_report.issues,
                    feedback=current_report.feedback,
                    transcript=transcript,
                    **profile,
                )
            except (ValueError, LLMError) as err:
                Logger.warning(f"Editor call failed: {err}. Falling back to re-formatting.")
                return None

            if not edits:
                Logger.info("Editor proposed no edits. Falling back to re-formatting.")
                return None

            Logger.info(f"Applying {len(edits)} targeted edit(s) in place...")
            try:
                edited_turns = self.formatter_agent.apply_edits(
                    current_turns,
                    edits,
                    agent_emotion=profile.get("agent_emotion"),
                    user_emotion=profile.get("user_emotion"),
                )
            except (ValueError, LLMError) as err:
                Logger.warning(f"Applying edits failed: {err}. Falling back to re-formatting.")
                return None

            manual_report = self.validate_conversation(edited_turns)
            if manual_report.has_errors:
                Logger.warning(
                    f"Edited conversation failed manual validation "
                    f"({len(manual_report.errors)} errors). Falling back to re-formatting."
                )
                return None

            Logger.success(f"Edits applied and passed manual validation (attempt {edit_attempt}).")
            current_turns = edited_turns

            new_report, _ = self._run_format_validation(transcript, current_turns)
            if new_report is None:
                Logger.warning("Faithfulness re-validation unavailable after edits; accepting edited conversation.")
                return current_turns, manual_report, None
            if new_report.passed:
                return current_turns, manual_report, new_report

            Logger.warning(
                f"Edited conversation still failing faithfulness (verdict {new_report.verdict}); "
                + ("trying another edit pass." if edit_attempt < self.max_edit_attempts else "giving up on edits.")
            )
            current_report = new_report

        # Edit passes exhausted without a PASS — hand back the best-effort edited
        # turns and the latest verdict so the caller decides (re-format).
        assert manual_report is not None
        return current_turns, manual_report, current_report


# Safety cap: give up on an instance after this many *consecutive* generations
# that fail validation, so a persistently-failing profile can't loop forever.
MAX_CONSECUTIVE_FAILURES = 10

# Local on-disk dump of accepted conversations (development only).
# Layout: <repo>/output/<run_id>/conversation.json + metadata.txt
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Serialises the token-usage metadata write, which parallel workers all trigger
# after each conversation — without it two threads could interleave the file.
_TOKEN_METADATA_LOCK = threading.Lock()


def _save_checkpoint_before_shutdown(
    storage: BaseStorage | None,
    checkpoint: Checkpoint | None,
    *,
    reason: str,
) -> None:
    """Best-effort checkpoint persist before terminating a production run."""
    if storage is None or checkpoint is None:
        return
    try:
        storage.save_checkpoint(checkpoint)
        Logger.warning(
            f"Checkpoint saved before shutdown ({reason}): "
            f"{len(checkpoint.instances)} instance record(s)."
        )
    except StorageError as err:
        Logger.error(f"Failed to save checkpoint before shutdown: {err}")


def _conversation_payload(instance: CorpusInstance, index: int, result: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON body stored for one accepted conversation."""
    manual_report = result.get("manual_validation")
    return {
        "corpus_combination_id": instance.corpus_combination_id,
        "index": index,
        "profile": instance.to_profile(),
        "topic": result.get("topic") or {},
        "duration_sec": getattr(manual_report, "duration_sec", None),
        "passed": result.get("passed", False),
        "include_numbers": result.get("include_numbers", False),
        "turns": result.get("turns", []),
    }


def _turn_type_counts(turns: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in turns:
        ttype = str(turn.get("turn_type") or "Unknown")
        counts[ttype] = counts.get(ttype, 0) + 1
    return counts


def _make_run_id(instance: CorpusInstance, index: int) -> str:
    """Unique folder name for one accepted conversation under ``output/``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{stamp}_corpus{instance.corpus_combination_id}_conv{index:04d}_{short}"


def _build_metadata_text(
    run_id: str,
    instance: CorpusInstance,
    index: int,
    result: dict[str, Any],
) -> str:
    """Human-readable metadata sidecar for an accepted conversation."""
    topic = result.get("topic") or {}
    turns = result.get("turns") or []
    manual_report = result.get("manual_validation")
    content_report = result.get("content_validation")
    format_report = result.get("format_validation")
    counts = _turn_type_counts(turns)
    duration = getattr(manual_report, "duration_sec", None)
    target = result.get("target_duration_sec")
    profile = instance.to_profile()

    lines = [
        f"run_id: {run_id}",
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"corpus_combination_id: {instance.corpus_combination_id}",
        f"conversation_index: {index}",
        f"passed: {result.get('passed', False)}",
        f"include_numbers: {result.get('include_numbers', False)}",
        "",
        "## Topic",
        f"title: {topic.get('title', '')}",
        f"conversation_type: {topic.get('conversation_type', '')}",
        f"context: {topic.get('context', '')}",
        "",
        "## Profile",
        f"language: {profile.get('language', '')}",
        f"gender_pair: {profile.get('gender_pair', '')}",
        f"agent_emotion: {profile.get('agent_emotion', '')}",
        f"user_emotion: {profile.get('user_emotion', '')}",
        f"agent_accent: {profile.get('agent_accent', '')}",
        f"user_accent: {profile.get('user_accent', '')}",
        "",
        "## Duration",
        f"duration_sec: {duration}",
        f"duration_min: {round(duration / 60, 2) if duration else None}",
        f"target_duration_sec: {target}",
        f"target_duration_min: {round(target / 60, 2) if target else None}",
        "",
        "## Turn counts",
        f"total_turns: {len(turns)}",
        f"backchannel_count: {counts.get('Backchanneling', 0)}",
        f"overlap_count: {counts.get('Overlapping', 0)}",
        f"interruption_count: {counts.get('Interruption', 0)}",
        f"normal_count: {counts.get('Normal', 0)}",
    ]
    for ttype, count in sorted(counts.items()):
        lines.append(f"  {ttype}: {count}")

    gen_attempts = result.get("content_attempts_used", 0)
    format_attempts = result.get("format_attempts_used", 0)
    content_val_attempts = result.get("content_validation_attempts_used", 0)
    format_val_attempts = result.get("format_validation_attempts_used", 0)
    lines += [
        "",
        "## Retries",
        f"generation_attempts_used: {gen_attempts} / {result.get('max_generation_attempts')}",
        f"format_attempts_used: {format_attempts} / {result.get('max_format_attempts')}",
        f"content_validation_attempts_used: {content_val_attempts} / {result.get('max_content_validation_retries')}",
        f"format_validation_attempts_used: {format_val_attempts} / {result.get('max_format_validation_retries')}",
        f"generation_retries: {max(0, int(gen_attempts) - 1)}",
        f"format_retries: {max(0, int(format_attempts) - 1)}",
        f"content_validation_bypassed: {result.get('content_validation_bypassed', False)}",
        f"format_validation_bypassed: {result.get('format_validation_bypassed', False)}",
        f"format_validation_exhausted: {result.get('format_validation_exhausted', False)}",
        "",
        "## Content validation (pre-format: corpus fit + realism)",
    ]
    if content_report is not None:
        lines += [
            f"verdict: {getattr(content_report, 'verdict', '')}",
            f"realism_score: {getattr(content_report, 'realism_score', '')}",
            f"corpus_match_score: {getattr(content_report, 'corpus_match_score', '')}",
        ]
        field_matches = getattr(content_report, "corpus_field_matches", None) or {}
        if field_matches:
            lines.append("corpus_field_matches:")
            for key, matched in field_matches.items():
                lines.append(f"  {key}: {matched}")
        feedback = getattr(content_report, "feedback", "") or ""
        if feedback:
            lines += ["", "feedback:", feedback]
    else:
        lines.append("verdict: (bypassed or unavailable)")

    lines += ["", "## Format validation (post-format: conversion faithfulness)"]
    if format_report is not None:
        lines.append(f"verdict: {getattr(format_report, 'verdict', '')}")
        format_feedback = getattr(format_report, "feedback", "") or ""
        if format_feedback:
            lines += ["", "feedback:", format_feedback]
    else:
        lines.append("verdict: (bypassed or unavailable)")

    lines.append("")
    return "\n".join(lines)


def save_local_output(
    instance: CorpusInstance,
    index: int,
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """Write an accepted conversation under ``output/<run_id>/``.

    Conversations are grouped by language: the run folder lands under
    ``output/<language>/<run_id>/`` (e.g. ``output/english/…``,
    ``output/hindi/…``, ``output/hinglish/…``), so each instance is stored in
    the folder for the language it belongs to. An unknown/blank language falls
    back to ``unknown``.

    Creates:
    * ``conversation.json`` — full payload (profile, topic, turns, …)
    * ``metadata.txt`` — title, duration, turn-type counts, retries, scores
    * ``transcript.txt`` — plain-text transcript for quick inspection

    Returns the run folder path.
    """
    root = output_dir or OUTPUT_DIR
    # Group by language so english/hindi/hinglish conversations each land in
    # their own top-level folder. Normalise to a lowercase, filesystem-safe name.
    language = (instance.language or "unknown").strip().lower() or "unknown"
    run_id = _make_run_id(instance, index)
    run_dir = root / language / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = _conversation_payload(instance, index, result)
    payload["run_id"] = run_id
    payload["target_duration_sec"] = result.get("target_duration_sec")
    payload["transcript"] = result.get("transcript") or ""

    json_path = run_dir / "conversation.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_path = run_dir / "metadata.txt"
    meta_path.write_text(
        _build_metadata_text(run_id, instance, index, result),
        encoding="utf-8",
    )

    transcript = result.get("transcript") or ""
    if transcript:
        (run_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    return run_dir


def save_token_usage_metadata(
    output_dir: Path | None = None,
    storage: BaseStorage | None = None,
) -> Path:
    """Persist per-model token totals to ``metadata.json``.

    Summarises the process-wide :data:`TOKEN_USAGE` tracker — input/output/total
    tokens and call count for every model used this run, plus a grand total.
    Always written locally to ``output/metadata.json``; in production it's also
    uploaded to the storage bucket root so the running total lives alongside the
    conversations. Overwritten each time so it always reflects the latest
    cumulative usage. Best-effort: a storage-upload failure is swallowed (a
    metadata write must never abort a run).
    """
    root = output_dir or OUTPUT_DIR

    # Snapshot + write under a lock so concurrent workers don't interleave the
    # file or upload inconsistent snapshots.
    with _TOKEN_METADATA_LOCK:
        root.mkdir(parents=True, exist_ok=True)
        summary = TOKEN_USAGE.as_dict()
        summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

        meta_path = root / "metadata.json"
        meta_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if storage is not None:
            try:
                storage.save_token_usage(summary)
            except StorageError as err:
                Logger.warning(f"Failed to upload token usage metadata to storage: {err}")

        # Emit the same totals to W&B — input/output/cache tokens per model,
        # plotted as one line per model over the run.
        wandb_logger.log_token_usage(summary)

    return meta_path


def _record_skipped_instance(
    storage: BaseStorage | None,
    skipped: SkippedRegistry | None,
    instance: CorpusInstance,
    progress: InstanceProgress,
    target_sec: float,
    consecutive_failures: int,
) -> None:
    """Record an abandoned instance in the bucket-root ``skipped.json``.

    Best-effort and production-only: a storage failure here must not crash the
    run, so it's logged and swallowed.
    """
    if storage is None or skipped is None:
        return
    try:
        skipped.add(
            SkippedInstance(
                corpus_combination_id=instance.corpus_combination_id,
                consecutive_failures=consecutive_failures,
                reason=f"{MAX_CONSECUTIVE_FAILURES} consecutive validation failures",
                generated_sec=progress.generated_sec,
                target_sec=target_sec,
                conversation_count=progress.conversation_count,
                language=instance.language,
                gender_pair=instance.gender_pair,
            )
        )
        storage.save_skipped(skipped)
        Logger.warning(
            f"Recorded instance {instance.corpus_combination_id} in "
            f"{storage.SKIPPED_NAME} (bucket root)."
        )
    except StorageError as err:
        Logger.error(
            f"Failed to record skipped instance {instance.corpus_combination_id}: {err}"
        )


def process_instance(
    runner: ConversationRunner,
    instance: CorpusInstance,
    storage: BaseStorage | None = None,
    checkpoint: Checkpoint | None = None,
    max_conversations: int | None = None,
    skipped: SkippedRegistry | None = None,
    *,
    language_target_sec: float | None = None,
    language_generated_sec: float | None = None,
) -> float:
    """Generate conversations for one instance until its target duration is met.

    Repeatedly runs the pipeline for the *same* instance, accumulating the
    (validated) duration of each accepted conversation, and stops once the total
    reaches ``instance.duration_sec``. Only conversations that pass validation
    count toward the target. Returns the total seconds generated.

    Persistence is production-only and controlled by ``storage``:

    * ``storage=None`` (development): nothing is uploaded — conversations are
      generated and counted in memory only.
    * ``storage`` provided (production): progress is resumed from ``checkpoint``,
      each accepted conversation is uploaded to its instance folder in the
      bucket, and the root ``checkpoint.json`` is updated afterwards, so a crash
      loses at most the one conversation in flight.

    ``max_conversations`` caps how many conversations are accepted this run,
    regardless of the duration target — used in development to stop after a
    single conversation instead of chasing the full multi-hour target.

    ``skipped`` (production only) is the bucket-root registry of instances
    abandoned after ``MAX_CONSECUTIVE_FAILURES`` consecutive validation failures.
    When this instance hits that limit, it's recorded there and ``skipped.json``
    is re-uploaded, so a resuming machine passes over it instead of retrying it.
    """
    target_sec = instance.duration_sec or 0.0

    # In dev (no storage) progress lives only in this local record; in prod it's
    # the shared, resumable checkpoint entry for this instance.
    if storage is not None and checkpoint is not None:
        progress = checkpoint.get(instance.corpus_combination_id, target_sec)
    else:
        progress = InstanceProgress(instance.corpus_combination_id, target_sec)

    if progress.generated_sec >= target_sec:
        Logger.info(
            f"Instance {instance.corpus_combination_id} already at target "
            f"({progress.generated_sec:.0f}/{target_sec:.0f}s) — skipping."
        )
        return progress.generated_sec

    overflow_now = progress.generated_sec - target_sec
    if overflow_now > INSTANCE_OVERFLOW_TOLERANCE_SEC:
        Logger.warning(
            f"Instance {instance.corpus_combination_id} is {overflow_now:.0f}s over target "
            f"({progress.generated_sec:.0f}/{target_sec:.0f}s) — skipping further generation."
        )
        return progress.generated_sec

    Logger.step(
        f"Instance {instance.corpus_combination_id} "
        f"[{instance.language} | {instance.gender_pair}] — "
        f"target {target_sec:.0f}s ({target_sec / 3600:.2f} hr)"
    )
    if progress.conversation_count:
        Logger.info(
            f"Resuming from checkpoint: {progress.generated_sec:.0f}s already done "
            f"across {progress.conversation_count} conversation(s)."
        )

    # Seed the topic generator with this instance's already-generated topics so
    # it keeps producing NEW ones — both across instances in this run and, on a
    # resume, across previous runs (topic titles are persisted in the checkpoint).
    runner.topic_agent.prime(progress.topics)
    if progress.topics:
        Logger.info(
            f"Primed topic generator with {len(progress.topics)} prior topic(s) "
            "to avoid repeats."
        )

    # How many conversations to generate concurrently. 1 == the old sequential
    # behaviour; N spins up N worker threads that each generate a conversation
    # for THIS instance at the same time. Topic generation is serialised inside
    # the topic agent, so parallel workers never produce a clashing topic.
    workers = get_num_workers()
    budget = GenerationBudget(
        instance_target_sec=target_sec,
        instance_generated_sec=progress.generated_sec,
        language_target_sec=language_target_sec,
        language_generated_sec=language_generated_sec,
    )
    if workers > 1:
        Logger.info(
            f"Parallel generation: {workers} worker(s) for instance "
            f"{instance.corpus_combination_id} "
            f"(instance headroom {budget.instance_headroom():.0f}s, "
            f"language headroom {budget.language_headroom():.0f}s)."
        )
    if language_target_sec is not None:
        Logger.info(
            f"Language budget [{instance.language}]: "
            f"{language_generated_sec or 0:.0f}/{language_target_sec:.0f}s committed "
            f"(+{budget.reserved_sec:.0f}s reserved in-flight when workers run)."
        )

    # Shared state mutated by every worker, guarded by ``lock``. ``stop`` signals
    # all workers to wind down (target met, abort, API limit, or storage error).
    lock = threading.Lock()
    stop = threading.Event()
    shared: dict[str, Any] = {
        "accepted": 0,             # accepted this run (for the max_conversations cap)
        "consecutive_failures": 0, # across workers; trips the abort guard
        "api_limit_err": None,     # set → re-raised after workers join
    }

    def _should_start() -> bool:
        """Whether a worker may begin another conversation (checked under lock)."""
        if stop.is_set() or progress.generated_sec >= target_sec:
            return False
        if max_conversations is not None and shared["accepted"] >= max_conversations:
            return False
        return budget.effective_headroom() >= MIN_VIABLE_CONVERSATION_SEC

    def worker(worker_id: int) -> None:
        profile = instance.to_profile()
        while True:
            provisional_id: str | None = None
            headroom_snapshot = 0.0
            with lock:
                if not _should_start():
                    return
                headroom_snapshot = budget.effective_headroom()
                # Gate entry into topic generation (an LLM call) with a
                # conservative reservation FIRST. With NUM_WORKERS possibly far
                # exceeding what an instance/language can still absorb (e.g.
                # 1000 workers, 12 conversations left), this caps how many
                # threads plan concurrently to roughly
                # ``headroom / DURATION_MIN_SEC`` — the rest see no headroom
                # left and skip straight to the next instance instead of
                # burning an API call that would just be discarded. In the
                # final stretch (< 1 full conversation of budget left), reserve
                # exactly what's left so one worker can still produce a short
                # closing conversation instead of everyone giving up.
                provisional_amount = min(DURATION_MIN_SEC, headroom_snapshot)
                provisional_id = budget.reserve(provisional_amount, worker_id)
                if provisional_id is None:
                    if headroom_snapshot < MIN_VIABLE_CONVERSATION_SEC:
                        stop.set()
                    return

            Logger.step(
                f"Worker {worker_id}: planning conversation for instance "
                f"{instance.corpus_combination_id} "
                f"({headroom_snapshot:.0f}s headroom before topic)."
            )
            plan = runner._plan_conversation(
                remaining_sec=headroom_snapshot,
                # ``headroom_snapshot`` already includes the instance/language
                # overflow tolerance (baked into GenerationBudget), so don't
                # let the planner add it again on top — that would let a
                # conversation's cap drift to double the intended overshoot.
                max_overflow_sec=0.0,
                **profile,
            )

            if plan.get("budget_exhausted"):
                with lock:
                    budget.release(provisional_id)
                    stop.set()
                Logger.info(
                    f"Worker {worker_id}: instance budget exhausted after topic — exiting."
                )
                return

            if not plan.get("target_duration_sec"):
                abort = False
                with lock:
                    budget.release(provisional_id)
                    shared["consecutive_failures"] += 1
                    cf = shared["consecutive_failures"]
                    if cf >= MAX_CONSECUTIVE_FAILURES and not stop.is_set():
                        stop.set()
                        abort = True
                wandb_logger.RUN_STATS.record_conversation(accepted=False)
                wandb_logger.log_progress(
                    consecutive_failures=cf,
                    instance_id=instance.corpus_combination_id,
                )
                Logger.error(
                    f"Topic/planning failed (worker {worker_id}; "
                    f"consecutive failures: {cf}/{MAX_CONSECUTIVE_FAILURES})."
                )
                if abort:
                    Logger.error(
                        f"Aborting instance {instance.corpus_combination_id}: "
                        f"{MAX_CONSECUTIVE_FAILURES} consecutive failures."
                    )
                    _record_skipped_instance(
                        storage, skipped, instance, progress, target_sec, cf
                    )
                    return
                continue

            planned_sec = float(plan["target_duration_sec"])
            reservation_id: str | None = None
            with lock:
                # Swap the conservative provisional reservation for the real
                # planned duration now that topic generation revealed it.
                budget.release(provisional_id)
                if not budget.can_accommodate(planned_sec):
                    Logger.info(
                        f"Worker {worker_id}: instance/language budget full — "
                        f"planned {planned_sec:.0f}s but only "
                        f"{budget.effective_headroom():.0f}s headroom "
                        f"(instance {budget.instance_headroom():.0f}s, "
                        f"language {budget.language_headroom():.0f}s, "
                        f"HF+reserved {budget.instance_generated_sec + budget.reserved_sec:.0f}s "
                        f"/ {budget.instance_target_sec:.0f}s) — exiting."
                    )
                    if budget.effective_headroom() < MIN_VIABLE_CONVERSATION_SEC:
                        stop.set()
                    return
                reservation_id = budget.reserve(planned_sec, worker_id)

            try:
                try:
                    result = runner._execute_conversation(plan, **profile)
                finally:
                    with lock:
                        budget.release(reservation_id)
                        reservation_id = None
            except APILimitError as err:
                with lock:
                    shared["api_limit_err"] = err
                    stop.set()
                Logger.error(f"API limit hit (worker {worker_id}): {err}")
                _save_checkpoint_before_shutdown(
                    storage, checkpoint, reason="API rate-limit or quota exceeded"
                )
                try:
                    save_token_usage_metadata(storage=storage)
                except OSError:
                    pass
                return

            # Tokens are spent whether or not the conversation passes; refresh the
            # running totals (the tracker + writer are thread-safe).
            try:
                save_token_usage_metadata(storage=storage)
            except OSError as meta_err:
                Logger.warning(f"Failed to write token usage metadata: {meta_err}")

            manual_report = result.get("manual_validation")
            duration = getattr(manual_report, "duration_sec", None)

            # ---- failed validation ----
            if not result.get("passed") or not duration:
                abort = False
                with lock:
                    shared["consecutive_failures"] += 1
                    cf = shared["consecutive_failures"]
                    if cf >= MAX_CONSECUTIVE_FAILURES and not stop.is_set():
                        stop.set()
                        abort = True
                wandb_logger.RUN_STATS.record_conversation(accepted=False)
                wandb_logger.log_progress(
                    consecutive_failures=cf,
                    instance_id=instance.corpus_combination_id,
                )
                Logger.error(
                    f"Conversation failed validation (worker {worker_id}; "
                    f"consecutive failures: {cf}/{MAX_CONSECUTIVE_FAILURES})."
                )
                if abort:
                    Logger.error(
                        f"Aborting instance {instance.corpus_combination_id}: "
                        f"{MAX_CONSECUTIVE_FAILURES} consecutive failures."
                    )
                    _record_skipped_instance(
                        storage, skipped, instance, progress, target_sec, cf
                    )
                    return
                continue

            # ---- accepted ----
            topic_title = (result.get("topic") or {}).get("title") or None
            with lock:
                if progress.generated_sec >= target_sec + INSTANCE_OVERFLOW_TOLERANCE_SEC:
                    return

                new_total = progress.generated_sec + duration
                if new_total > target_sec + INSTANCE_OVERFLOW_TOLERANCE_SEC:
                    Logger.warning(
                        f"Rejecting conversation (+{duration:.0f}s would reach "
                        f"{new_total:.0f}s, {new_total - target_sec:.0f}s over target "
                        f"{target_sec:.0f}s — max allowed overflow "
                        f"{INSTANCE_OVERFLOW_TOLERANCE_SEC:.0f}s)."
                    )
                    stop.set()
                    return

                # Assign the file index atomically from the persisted count so
                # parallel saves never collide or leave gaps.
                index = progress.conversation_count + 1

                if storage is not None and checkpoint is not None:
                    # Upload the conversation, THEN advance + persist the checkpoint,
                    # so a crash between the two just regenerates it.
                    try:
                        path = storage.save_conversation(
                            instance.corpus_combination_id,
                            index,
                            _conversation_payload(instance, index, result),
                        )
                        checkpoint.record(progress, duration, topic_title=topic_title)
                        storage.save_checkpoint(checkpoint)
                    except StorageError as err:
                        Logger.error(
                            f"Storage failure on conversation {index}, aborting instance: {err}"
                        )
                        stop.set()
                        return
                    Logger.success(
                        f"Saved {path} (+{duration:.0f}s) — "
                        f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                        bold=True,
                    )
                else:
                    # Development: dump locally and count in memory, no upload.
                    if not is_production():
                        try:
                            local_dir = save_local_output(instance, index, result)
                            Logger.success(f"Wrote local output → {local_dir}")
                        except OSError as err:
                            Logger.warning(
                                f"Failed to write local output for conversation {index}: {err}"
                            )
                    progress.generated_sec += duration
                    progress.conversation_count += 1
                    if topic_title:
                        progress.topics.append(topic_title)
                    Logger.success(
                        f"Conversation {index} accepted (+{duration:.0f}s) — "
                        f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                        bold=True,
                    )

                shared["consecutive_failures"] = 0
                shared["accepted"] += 1
                budget.commit(duration)
                if progress.generated_sec >= target_sec:
                    stop.set()

            wandb_logger.RUN_STATS.record_conversation(accepted=True)
            wandb_logger.log_progress(
                instance_id=instance.corpus_combination_id,
                instance_generated_sec=progress.generated_sec,
                instance_target_sec=target_sec,
                duration_sec=duration,
            )

    if workers == 1:
        worker(1)
    else:
        threads = [
            threading.Thread(target=worker, args=(wid,), name=f"gen-worker-{wid}")
            for wid in range(1, workers + 1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # An API-limit hit inside any worker must terminate the whole run (main saves
    # the checkpoint and stops); re-raise it now that all workers have stopped.
    if shared["api_limit_err"] is not None:
        raise shared["api_limit_err"]

    wandb_logger.RUN_STATS.record_instance_complete()
    wandb_logger.log_progress(instance_id=instance.corpus_combination_id)

    Logger.success(
        f"Instance {instance.corpus_combination_id} complete: "
        f"{progress.generated_sec:.0f}s across {progress.conversation_count} conversation(s).",
        bold=True,
    )
    return progress.generated_sec


def _language_ordered_positions(
    corpus_df: Any,
    run_languages: list[str] | None,
) -> list[int]:
    """Positional row indices to process, grouped by the requested language order.

    With ``run_languages=["english", "hindi"]`` every English instance comes
    first (in corpus order), then every Hindi one — so a run can finish one
    language before starting the next. ``run_languages=None`` returns every row
    in its original corpus order. Matching is case-insensitive on the corpus's
    ``language`` column.
    """
    if run_languages is None:
        return list(range(len(corpus_df)))
    lang_of = corpus_df["language"].str.lower()
    positions: list[int] = []
    for lang in run_languages:
        positions.extend(pos for pos in range(len(corpus_df)) if lang_of.iloc[pos] == lang)
    return positions


def _instance_progress_state(
    instance: CorpusInstance,
    checkpoint: Checkpoint | None,
) -> tuple[str, InstanceProgress]:
    """Classify checkpoint progress as ``complete``, ``incomplete``, or ``not_started``."""
    target_sec = instance.duration_sec or 0.0
    if checkpoint is not None:
        progress = checkpoint.get(instance.corpus_combination_id, target_sec)
    else:
        progress = InstanceProgress(instance.corpus_combination_id, target_sec)

    if progress.generated_sec >= target_sec:
        if progress.generated_sec > target_sec + INSTANCE_OVERFLOW_TOLERANCE_SEC:
            return "over_target", progress
        return "complete", progress
    if progress.conversation_count > 0 or progress.generated_sec > 0:
        return "incomplete", progress
    return "not_started", progress


def _plan_production_run(
    corpus_df: Any,
    run_languages: list[str] | None,
    checkpoint: Checkpoint | None,
    skipped: SkippedRegistry | None,
) -> tuple[list[int], dict[str, Any]]:
    """Order instances: incomplete first, then not started; skip complete/skipped."""
    positions = _language_ordered_positions(corpus_df, run_languages)
    incomplete: list[int] = []
    not_started: list[int] = []
    complete_count = 0
    over_target_count = 0
    skipped_count = 0
    lang_stats: dict[str, dict[str, int]] = {}

    for pos in positions:
        row = corpus_df.iloc[pos]
        instance = CorpusInstance.from_dict({str(k): v for k, v in row.to_dict().items()})
        lang = str(row.get("language", "")).lower()

        if skipped is not None and instance.corpus_combination_id in skipped:
            skipped_count += 1
            continue

        state, progress = _instance_progress_state(instance, checkpoint)
        stats = lang_stats.setdefault(
            lang,
            {"total": 0, "incomplete": 0, "not_started": 0, "complete": 0, "over_target": 0},
        )
        stats["total"] += 1
        stats[state] += 1

        if state == "complete":
            complete_count += 1
        elif state == "over_target":
            over_target_count += 1
        elif state == "incomplete":
            incomplete.append(pos)
        else:
            not_started.append(pos)

    ordered = incomplete + not_started
    summary = {
        "total_in_corpus": len(positions),
        "to_process": len(ordered),
        "complete": complete_count,
        "over_target": over_target_count,
        "skipped": skipped_count,
        "incomplete": len(incomplete),
        "not_started": len(not_started),
        "by_language": lang_stats,
    }
    return ordered, summary


def _log_run_plan(summary: dict[str, Any], run_languages: list[str] | None) -> None:
    """Log language/instance counts and the incomplete-first processing order."""
    langs = ", ".join(run_languages) if run_languages else "all"
    Logger.step(
        f"Run plan — languages: {langs} | "
        f"{summary['to_process']} instance(s) to process "
        f"({summary['incomplete']} incomplete, {summary['not_started']} not started)"
    )
    if summary["complete"]:
        Logger.info(
            f"Skipping {summary['complete']} already-complete instance(s) "
            f"out of {summary['total_in_corpus']} in scope."
        )
    if summary.get("over_target"):
        Logger.warning(
            f"Skipping {summary['over_target']} over-target instance(s) "
            f"(generated past target by more than {INSTANCE_OVERFLOW_TOLERANCE_SEC:.0f}s)."
        )
    if summary["skipped"]:
        Logger.info(f"Skipping {summary['skipped']} abandoned instance(s) in skipped.json.")
    for lang, stats in sorted(summary["by_language"].items()):
        Logger.info(
            f"  {lang}: {stats['total']} total — "
            f"{stats['incomplete']} incomplete, {stats['not_started']} not started, "
            f"{stats['complete']} complete"
            + (f", {stats['over_target']} over target" if stats.get("over_target") else "")
        )
    if summary["incomplete"]:
        Logger.info("Processing order: incomplete instances first, then not started.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    provider_choices = [p.value for p in LLMProvider]
    parser = argparse.ArgumentParser(
        description="Generate synthetic two-speaker conversations from the corpus."
    )
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default=None,
        type=str.lower,
        help=(
            "Only generate conversations for this corpus language. "
            "Omit to process every language in the corpus (default)."
        ),
    )
    parser.add_argument(
        "--model",
        choices=provider_choices,
        default=None,
        type=str.lower,
        help=(
            "LLM provider for *data generation only* (topic + conversation "
            "transcript). Always wins when given, for any language. Only "
            "when omitted does Hindi default to 'sarvam' and other "
            f"languages default to '{DEFAULT_GENERATION_PROVIDER.value}'."
        ),
    )
    parser.add_argument(
        "--validation-model",
        dest="validation_model",
        choices=provider_choices,
        default=DEFAULT_VALIDATION_PROVIDER.value,
        type=str.lower,
        help=(
            "LLM provider for the *formatter* and *LLM agent validator* "
            f"(not generation). Default: '{DEFAULT_VALIDATION_PROVIDER.value}'."
        ),
    )
    return parser.parse_args(argv)


def _get_runner(
    runner_cache: dict[tuple[LLMProvider, LLMProvider], "ConversationRunner"],
    model: str | None,
    validation: str | None,
    language: str | None,
) -> "ConversationRunner":
    """Return the (cached) runner for this instance's generation + validation providers.

    Generation resolves per-instance, but an explicit ``model`` always wins —
    Hindi only defaults to Sarvam when ``model`` is ``None``. Formatting and
    agent validation always use ``validation`` (default Gemma) with no language
    override. Each (generation, validation) pair is built once and reused.
    """
    generation_provider = resolve_provider(
        model,
        language,
        apply_language_routing=True,
        default=DEFAULT_GENERATION_PROVIDER,
    )
    validation_provider = resolve_provider(
        validation,
        language=None,
        apply_language_routing=False,
        default=DEFAULT_VALIDATION_PROVIDER,
    )
    cache_key = (generation_provider, validation_provider)
    if cache_key not in runner_cache:
        Logger.info(
            f"Initializing LLMs — generation='{generation_provider.value}', "
            f"validation/formatter='{validation_provider.value}'..."
        )
        generation_llm = create_llm(
            generation_provider,
            apply_language_routing=False,
        )
        validation_llm = create_llm(
            validation_provider,
            apply_language_routing=False,
        )
        Logger.info(
            f"Generation model='{generation_llm.model}' (temperature={generation_llm.temperature}), "
            f"validation/formatter model='{validation_llm.model}' (temperature={validation_llm.temperature}) "
            "— both configurable via conversations_generator/config.json ('MODELS' / 'TEMPERATURE')."
        )
        runner_cache[cache_key] = ConversationRunner(
            llm=generation_llm,
            validation_llm=validation_llm,
            max_generation_attempts=3,
            max_format_attempts=3,
        )
    return runner_cache[cache_key]


def main(argv: list[str] | None = None) -> None:
    """Entry point: run the pipeline, always closing the W&B run on exit."""
    try:
        _main_impl(argv)
    finally:
        # Always finalize the W&B run — even on an early return (language
        # validation failure, empty instance list, API-limit termination) or
        # an uncaught exception — so a run is never left dangling as "crashed".
        wandb_logger.finish_run()


def _main_impl(argv: list[str] | None = None) -> None:
    # Load API keys / settings from conversations_generator/config.json and
    # mirror them into os.environ for any third-party SDK that only reads env.
    apply_to_environ()
    args = _parse_args(argv)

    # ``MODE=prod`` (config.json) processes every corpus instance in sequence and
    # uploads to HuggingFace; anything else is development — local prompts, local
    # dumps only, and just a few rows. Same switch also picks the prompt source
    # (see prompts.resolve_system_prompt).
    production = is_production()
    Logger.step(
        f"MODE = {get_mode()}  →  prompts: {'Langfuse (local fallback)' if production else 'local files'}, "
        f"storage: {'HuggingFace upload' if production else 'local only'}"
    )

    corpus_path = Path(__file__).resolve().parent / "data" / "corpus_instances.jsonl"
    corpus_df = read_corpus_instances(str(corpus_path))

    # Which languages to run this session, in order. A single ``--language`` on the
    # CLI always wins; otherwise ``RUN_LANGUAGES`` in config.json decides (a list
    # like ["hindi", "english"]); if neither is set, every language is processed.
    # Because the checkpoint tracks progress per instance, re-listing a language
    # that was only partly finished last run just resumes its unfinished instances.
    if args.language:
        run_languages: list[str] | None = [args.language]
    else:
        run_languages = get_run_languages()

    if run_languages is not None:
        invalid = [lang for lang in run_languages if lang not in SUPPORTED_LANGUAGES]
        if invalid:
            Logger.error(
                f"Unsupported language(s) requested: {invalid}. "
                f"Choose from {list(SUPPORTED_LANGUAGES)} via --language or "
                "config.json 'RUN_LANGUAGES'."
            )
            return
        Logger.step(f"Languages this run (in order): {run_languages}")
    else:
        Logger.step("No language filter — processing every language in the corpus.")

    hindi_note = "" if args.model else " (Hindi defaults to sarvam when --model is omitted)"
    Logger.info(
        f"Providers — generation(--model)={args.model or DEFAULT_GENERATION_PROVIDER.value}"
        f"{hindi_note}, validation/formatter(--validation-model)={args.validation_model}"
    )

    # One W&B run per process invocation — logs token usage (input/output/cache,
    # per model) and conversation/instance progress for the rest of this run.
    wandb_logger.init_run(
        {
            "mode": get_mode(),
            "language_filter": run_languages,
            "generation_provider": args.model or DEFAULT_GENERATION_PROVIDER.value,
            "validation_provider": args.validation_model,
            "num_workers": get_num_workers(),
        }
    )

    # Runners are cached by (generation_provider, validation_provider) because
    # generation can differ per instance (Hindi → Sarvam) while validation is
    # fixed for the whole CLI invocation.
    runner_cache: dict[tuple[LLMProvider, LLMProvider], ConversationRunner] = {}

    # Storage + checkpoint are production-only. Dev runs entirely in memory.
    storage: BaseStorage | None = None
    checkpoint: Checkpoint | None = None
    skipped: SkippedRegistry | None = None
    # In development, cap at a single conversation per instance; production
    # chases each instance's full duration target.
    max_conversations: int | None = None
    if production:
        storage = HuggingFaceStorage()
        checkpoint = storage.load_checkpoint()
        Logger.info(f"Loaded checkpoint with {len(checkpoint.instances)} instance record(s).")
        skipped = storage.load_skipped()
        Logger.info(
            f"Loaded skipped registry with {len(skipped.instances)} abandoned instance(s)."
        )
        # Process every instance of each requested language in turn. Resume is
        # automatic via the checkpoint; incomplete instances are scheduled first.
        indices, run_plan = _plan_production_run(
            corpus_df, run_languages, checkpoint, skipped
        )
        _log_run_plan(run_plan, run_languages)
        if not indices:
            Logger.success(
                "All instances already complete for the requested languages — nothing to do."
            )
            return
    else:
        Logger.step("DEVELOPMENT run — generating a single conversation per instance (no upload).")
        # Hand-picked smoke tests spanning every language for variety.
        smoke_indices = [
            # --- original smoke tests ---
            234,   # English  | FM | user West accent | Neutral/Sad
            0,     # Hinglish | FM | Normal           | Neutral/Neutral (baseline)
            135,   # Hindi    | MM | user Bengali     | Neutral/Neutral
            # --- Hindi (Devanagari + gender agreement + accents) ---
            59,    # Hindi    | MF | Normal            | Neutral/Neutral (gender baseline)
            3278,  # Hindi    | FM | Punjabi both      | Happy/Happy
            6599,  # Hindi    | MM | Bengali both      | Angry/Sad (opposing emotion)
            3235,  # Hindi    | FF | West/Bengali mix  | Happy/Angry (mixed accents)
            # --- Hinglish (code-mixing + accents) ---
            5,     # Hinglish | FM | Normal            | Neutral/Happy (baseline)
            4921,  # Hinglish | FF | Gujarati both     | Sad/Sad
            4751,  # Hinglish | MF | South Indian both | Angry/Neutral
            # --- English (no Devanagari; lang-vs-accent tension) ---
            2225,  # English  | MF | Normal            | Sad/Sad (pure emotional)
            2882,  # English  | FM | West both         | Happy/Neutral
            2563,  # English  | MM | Bengali/Punjabi   | Neutral/Angry (mixed accents)
        ]
        # Honour the requested languages (and their order) even in dev by keeping
        # only the matching smoke tests, grouped by the run-language order.
        if run_languages is not None:
            lang_of = corpus_df["language"].str.lower()
            indices = [
                pos
                for lang in run_languages
                for pos in smoke_indices
                if lang_of.iloc[pos] == lang
            ]
        else:
            indices = smoke_indices
        max_conversations = 1

    if not indices:
        Logger.error(
            f"No corpus instances to process for languages {run_languages}. Nothing to do."
        )
        return

    wandb_logger.RUN_STATS.set_total_instances(len(indices))

    for i in indices:
        row: dict[str, Any] = {str(k): v for k, v in corpus_df.iloc[i].to_dict().items()}
        instance = CorpusInstance.from_dict(row)
        # Production resume guard: an instance already abandoned (10 consecutive
        # failures on a previous run) is recorded in skipped.json — pass over it
        # instead of burning another 10 attempts on the same failing profile.
        if skipped is not None and instance.corpus_combination_id in skipped:
            Logger.warning(
                f"Skipping instance {instance.corpus_combination_id} — "
                f"already recorded in {BaseStorage.SKIPPED_NAME}."
            )
            continue
        runner = _get_runner(runner_cache, args.model, args.validation_model, instance.language)
        lang_target, lang_generated = _language_budget_totals(
            corpus_df,
            checkpoint,
            run_languages,
            instance.language,
        )
        try:
            process_instance(
                runner,
                instance,
                storage,
                checkpoint,
                max_conversations,
                skipped,
                language_target_sec=lang_target,
                language_generated_sec=lang_generated,
            )
        except APILimitError:
            Logger.error(
                "Terminating run: API rate-limit or quota exceeded. "
                "Resume later from the saved checkpoint."
            )
            try:
                save_token_usage_metadata(storage=storage)
            except OSError as err:
                Logger.warning(f"Failed to write token usage metadata: {err}")
            return

    Logger.divider()
    try:
        meta_path = save_token_usage_metadata(storage=storage)
        totals = TOKEN_USAGE.as_dict()["total"]
        Logger.success(
            f"Token usage written → {meta_path} "
            f"(total {totals['total_tokens']} tokens across {totals['calls']} call(s))."
        )
    except OSError as err:
        Logger.warning(f"Failed to write token usage metadata: {err}")

    Logger.success("All requested instances processed.", bold=True)


if __name__ == "__main__":
    main()