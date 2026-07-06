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

    python -m conversations_generator.runner --language=english --corpus-size=120 --workers=4

``--language`` restricts the run to one corpus language (``hindi`` / ``hinglish``
/ ``english``); omit it to process every language, as before.

``--corpus-size`` caps total generated audio duration (minutes) for the filtered
language(s). Generation stops once accepted conversations reach that total.

    python -m conversations_generator.runner --tokenstats --language=english

``--tokenstats`` scans the HuggingFace bucket and prints input/output token
summaries from stored conversation JSON (no generation run).

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
    get_number_inclusion_percentage,
    is_production,
)
from .llm import (
    DEFAULT_GENERATION_PROVIDER,
    DEFAULT_VALIDATION_PROVIDER,
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
from .usage_tracker import UsageTracker, usage_context
from .token_stats import (
    ModelInfo,
    legacy_krutrim_models,
    models_metadata_lines,
    patch_hf_model_metadata,
    print_token_stats,
)
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


class CorpusBudget:
    """Thread-safe cap on total accepted duration for a language-filtered run."""

    def __init__(self, target_sec: float | None, initial_sec: float = 0.0) -> None:
        self.target_sec = target_sec
        self._generated_sec = initial_sec
        self._lock = threading.Lock()

    @property
    def generated_sec(self) -> float:
        with self._lock:
            return self._generated_sec

    def can_continue(self) -> bool:
        if self.target_sec is None:
            return True
        with self._lock:
            return self._generated_sec < self.target_sec

    def remaining_sec(self) -> float | None:
        if self.target_sec is None:
            return None
        with self._lock:
            return max(0.0, self.target_sec - self._generated_sec)

    def add(self, duration_sec: float) -> None:
        with self._lock:
            self._generated_sec += duration_sec


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
    def generate_topic(
        self,
        usage_tracker: UsageTracker,
        **profile: Any,
    ) -> dict[str, str]:
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
                with usage_context(usage_tracker, stage="topic", attempt=attempt):
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
    def run(self, **profile: Any) -> dict[str, Any]:
        """Run the full pipeline with a content gate before formatting.

        Flow::

            topic = generate_topic()

            # Stage 2+3: generate a transcript and CONTENT-validate it. The
            # transcript is only handed on once the content judge PASSES.
            for gen_attempt in range(max_generation_attempts):
                transcript = generator_agent(...)
                if content_validator(transcript) PASSES: break
                # else feed the content feedback + previous transcript back to
                # the GENERATOR and regenerate
            else:
                return failure  # never passed content validation → not formatted

            # Stage 4: format the APPROVED transcript, then validate the
            # formatting only (deterministic timing + LLM faithfulness).
            for fmt_attempt in range(max_format_attempts):
                turns = formatter_agent(transcript, ...)
                if manual_validation(turns) fails: re-format; continue
                if format_validator(transcript, turns) PASSES: done
                # else the formatter was unfaithful → re-format (transcript kept)

        So content quality is fixed by **regeneration** before formatting, and the
        post-format judge only checks faithful conversion — it never re-judges
        accent/emotion/realism, which are already approved.

        Returns
        -------
        dict with keys: topic, transcript, turns, manual_validation,
            content_validation, format_validation, and retry/bypass metadata.
        """
        Logger.step(f"Starting pipeline for language: {profile.get('language', 'unknown')}")

        usage_tracker = UsageTracker()

        # Decide ONCE per conversation whether this one is number-rich, drawn from
        # NUMBER_INCLUSION_PERCENTAGE (default 50%). Fixed for the whole pipeline
        # (topic + every regeneration) so the topic and dialogue stay aligned.
        include_numbers = random.random() < get_number_inclusion_percentage()
        Logger.info(
            f"Number inclusion: {'ON — numbers + reasoning' if include_numbers else 'OFF — qualitative'}"
        )

        try:
            topic = self.generate_topic(usage_tracker, include_numbers=include_numbers, **profile)
        except (ValueError, LLMError) as err:
            # Topic generation exhausted its retries (e.g. the model kept returning
            # empty/invalid JSON). Fail THIS conversation gracefully instead of
            # crashing the whole run — process_instance counts it and moves on.
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
                "usage": usage_tracker.to_dict(),
                "passed": False,
            }

        # Draw one exact target duration for this conversation from the Gaussian
        # so lengths follow the intended 4–8 min distribution. Sampled once and
        # reused across all retries so every regeneration aims for the same target.
        target_duration_sec = sample_target_duration_sec()
        Logger.info(
            f"Target conversation duration: {target_duration_sec:.1f}s "
            f"({target_duration_sec / 60:.2f} min)"
        )

        transcript: str = ""
        content_report: ContentValidationReport | None = None
        content_validation_bypassed = False
        content_attempts_used = 0
        content_validation_attempts_used = 0

        # Feedback threaded back to the GENERATOR across regeneration attempts.
        generator_feedback: str | None = None
        previous_transcript: str | None = None

        # ---- Stage 2+3: generate a transcript and gate on content validation ----
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
                with usage_context(
                    usage_tracker,
                    stage="conversation_generation",
                    attempt=gen_attempt,
                ):
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
                transcript, topic, profile, usage_tracker, gen_attempt
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
                "usage": usage_tracker.to_dict(),
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
                with usage_context(
                    usage_tracker,
                    stage="formatting",
                    attempt=fmt_attempt,
                ):
                    turns = self.formatter_agent.run(
                        transcript=transcript,
                        agent_emotion=profile.get("agent_emotion"),
                        user_emotion=profile.get("user_emotion"),
                        language=profile.get("language"),
                        feedback=formatter_feedback,
                        previous_output=previous_output,
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
                transcript, turns, usage_tracker, fmt_attempt
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
            edit_result = self._repair_by_editing(
                turns, transcript, format_report, profile, usage_tracker
            )
            if edit_result is not None:
                turns, edited_manual, edited_format = edit_result
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
            "usage": usage_tracker.to_dict(),
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

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _run_content_validation(
        self,
        transcript: str,
        topic: dict[str, str],
        profile: dict[str, Any],
        usage_tracker: UsageTracker,
        generation_attempt: int,
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
                with usage_context(
                    usage_tracker,
                    stage="content_validation",
                    attempt=generation_attempt * 100 + attempt,
                ):
                    report = self.content_validator.run(
                        transcript=transcript, topic=topic, **profile
                    )
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
        usage_tracker: UsageTracker,
        format_attempt: int,
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
                with usage_context(
                    usage_tracker,
                    stage="format_validation",
                    attempt=format_attempt * 100 + attempt,
                ):
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
        usage_tracker: UsageTracker,
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
                with usage_context(
                    usage_tracker,
                    stage="editor",
                    attempt=edit_attempt,
                ):
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

            new_report, _ = self._run_format_validation(
                transcript, current_turns, usage_tracker, edit_attempt
            )
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

# tqdm bar format: percentage, bar, counts, elapsed, ETA, and rate.
_DURATION_BAR_FMT = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n:.0f}/{total:.0f}s [{elapsed}<{remaining}, {rate_fmt}]"
)
_INSTANCE_BAR_FMT = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
)


def _format_audio_minutes(sec: float) -> str:
    """Compact duration label for tqdm postfixes."""
    if sec >= 3600:
        return f"{sec / 3600:.2f}h"
    return f"{sec / 60:.1f}m"


def _corpus_postfix(corpus_budget: CorpusBudget | None, corpus_size_min: float | None) -> str:
    if corpus_budget is None:
        return ""
    generated = _format_audio_minutes(corpus_budget.generated_sec)
    if corpus_size_min is not None:
        return f"audio {generated}/{corpus_size_min:.1f}m"
    return f"audio {generated}"

# Local on-disk dump of every accepted conversation (dev and prod).
# Layout: <repo>/output/<run_id>/conversation.json + metadata.txt
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _conversation_payload(
    instance: CorpusInstance,
    index: int,
    result: dict[str, Any],
    *,
    models: ModelInfo | None = None,
) -> dict[str, Any]:
    """Build the JSON body stored for one accepted conversation."""
    manual_report = result.get("manual_validation")
    payload: dict[str, Any] = {
        "corpus_combination_id": instance.corpus_combination_id,
        "index": index,
        "profile": instance.to_profile(),
        "topic": result.get("topic") or {},
        "duration_sec": getattr(manual_report, "duration_sec", None),
        "passed": result.get("passed", False),
        "include_numbers": result.get("include_numbers", False),
        "turns": result.get("turns", []),
        "usage": result.get("usage"),
    }
    model_data = models.to_dict() if models else result.get("models")
    if model_data:
        payload["models"] = model_data
    return payload


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
    ]
    models = result.get("models")
    if models:
        lines += models_metadata_lines(models)

    lines += [
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

    usage = result.get("usage")
    if usage:
        lines += _usage_metadata_lines(usage)

    lines.append("")
    return "\n".join(lines)


def _usage_metadata_lines(usage: dict[str, Any]) -> list[str]:
    """Render usage dict into metadata.txt lines."""
    totals = usage.get("totals") or {}
    lines = [
        "",
        "## LLM usage",
        f"total_calls: {totals.get('calls', 0)}",
        f"total_input_tokens: {totals.get('input_tokens', 0)}",
        f"total_output_tokens: {totals.get('output_tokens', 0)}",
        f"total_tokens: {totals.get('total_tokens', 0)}",
        f"total_llm_duration_sec: {totals.get('duration_sec', 0)}",
        "",
        "### By agent",
    ]
    by_agent = usage.get("by_agent") or {}
    for agent, stats in sorted(by_agent.items()):
        lines.append(
            f"{agent}: calls={stats.get('calls', 0)}, "
            f"in={stats.get('input_tokens', 0)}, out={stats.get('output_tokens', 0)}, "
            f"total={stats.get('total_tokens', 0)}, "
            f"duration_sec={stats.get('duration_sec', 0)}"
        )
    lines += ["", "### Per call (chronological)"]
    for call in usage.get("calls") or []:
        attempt = call.get("attempt")
        attempt_s = f", attempt={attempt}" if attempt is not None else ""
        lines.append(
            f"- [{call.get('stage', '')}{attempt_s}] {call.get('agent', '')} "
            f"({call.get('model', '')}): in={call.get('input_tokens', 0)}, "
            f"out={call.get('output_tokens', 0)}, "
            f"duration_sec={call.get('duration_sec', 0)}"
        )
    return lines


def save_local_output(
    instance: CorpusInstance,
    index: int,
    result: dict[str, Any],
    output_dir: Path | None = None,
    *,
    models: ModelInfo | None = None,
) -> Path:
    """Write an accepted conversation under ``output/<run_id>/``.

    Creates:
    * ``conversation.json`` — full payload (profile, topic, turns, …)
    * ``metadata.txt`` — title, duration, turn-type counts, retries, scores
    * ``transcript.txt`` — plain-text transcript for quick inspection

    Returns the run folder path.
    """
    root = output_dir or OUTPUT_DIR
    run_id = _make_run_id(instance, index)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = _conversation_payload(instance, index, result, models=models)
    payload["run_id"] = run_id
    payload["target_duration_sec"] = result.get("target_duration_sec")
    payload["transcript"] = result.get("transcript") or ""
    if models:
        result["models"] = models.to_dict()

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


def process_instance(
    runner: ConversationRunner,
    instance: CorpusInstance,
    storage: BaseStorage | None = None,
    checkpoint_caches: dict[str, Checkpoint] | None = None,
    skipped_caches: dict[str, SkippedRegistry] | None = None,
    max_conversations: int | None = None,
    corpus_budget: CorpusBudget | None = None,
    storage_lock: threading.Lock | None = None,
    models: ModelInfo | None = None,
) -> float:
    """Generate conversations for one instance until its target duration is met.

    Repeatedly runs the pipeline for the *same* instance, accumulating the
    (validated) duration of each accepted conversation, and stops once the total
    reaches ``instance.duration_sec``. Only conversations that pass validation
    count toward the target. Returns the total seconds generated.

    Persistence is production-only and controlled by ``storage``:

    * ``storage=None`` (development): nothing is uploaded — conversations are
      generated and counted in memory only.
    * ``storage`` provided (production): progress is resumed from the language
      folder's ``checkpoint.json``, each accepted conversation is uploaded under
      ``<language>/instance_<id>/conversation_<n>/``, and the checkpoint is
      updated afterwards, so a crash loses at most the one conversation in flight.

    ``max_conversations`` caps how many conversations are accepted this run,
    regardless of the duration target — used in development to stop after a
    single conversation instead of chasing the full multi-hour target.

    ``skipped`` (production only) is the per-language registry of instances
    abandoned after ``MAX_CONSECUTIVE_FAILURES`` consecutive validation failures.
    """
    target_sec = instance.duration_sec or 0.0
    lang_key = BaseStorage.normalize_language(instance.language)

    checkpoint: Checkpoint | None = None
    skipped: SkippedRegistry | None = None
    if storage is not None:
        lock = storage_lock or threading.Lock()
        with lock:
            if checkpoint_caches is not None:
                if lang_key not in checkpoint_caches:
                    checkpoint_caches[lang_key] = storage.load_checkpoint(instance.language)
                checkpoint = checkpoint_caches[lang_key]
            if skipped_caches is not None:
                if lang_key not in skipped_caches:
                    skipped_caches[lang_key] = storage.load_skipped(instance.language)
                skipped = skipped_caches[lang_key]

    # In dev (no storage) progress lives only in this local record; in prod it's
    # the shared, resumable checkpoint entry for this instance.
    if checkpoint is not None:
        progress = checkpoint.get(instance.corpus_combination_id, target_sec)
    else:
        progress = InstanceProgress(instance.corpus_combination_id, target_sec)

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

    # Continue numbering after whatever's already recorded so we never overwrite
    # a conversation a previous machine uploaded.
    index = progress.conversation_count
    consecutive_failures = 0
    accepted = 0  # conversations accepted this run (for the max_conversations cap)

    duration_bar = tqdm(
        total=target_sec,
        initial=progress.generated_sec,
        desc=(
            f"Inst {instance.corpus_combination_id} "
            f"[{instance.language} | {instance.gender_pair}]"
        ),
        unit="s",
        dynamic_ncols=True,
        leave=False,
        bar_format=_DURATION_BAR_FMT,
    )

    while progress.generated_sec < target_sec:
        if corpus_budget is not None and not corpus_budget.can_continue():
            Logger.info(
                f"Corpus size target reached ({corpus_budget.generated_sec / 60:.2f} min); "
                f"stopping instance {instance.corpus_combination_id}."
            )
            break
        if max_conversations is not None and accepted >= max_conversations:
            break
        index += 1
        duration_bar.set_postfix_str(
            f"conv {index}, accepted {accepted}, fails {consecutive_failures}",
            refresh=False,
        )
        Logger.divider()
        Logger.info(
            f"Instance {instance.corpus_combination_id} — conversation {index} "
            f"(progress {progress.generated_sec:.0f}/{target_sec:.0f}s, "
            f"{progress.generated_sec / target_sec * 100 if target_sec else 0:.1f}%)"
        )

        result = runner.run(**instance.to_profile())
        if models:
            result["models"] = models.to_dict()
        manual_report = result.get("manual_validation")
        duration = getattr(manual_report, "duration_sec", None)

        if not result.get("passed") or not duration:
            consecutive_failures += 1
            Logger.error(
                f"Conversation {index} failed validation "
                f"(consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})."
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                Logger.error(
                    f"Aborting instance {instance.corpus_combination_id}: "
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive failures."
                )
                # Production only: record the abandoned instance in the bucket's
                # root skipped.json so it's audited and skipped on resume instead
                # of burning another 10 attempts on it. Best-effort — a storage
                # failure here must not crash the whole run.
                if storage is not None and skipped is not None:
                    try:
                        lock = storage_lock or threading.Lock()
                        with lock:
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
                            storage.save_skipped(skipped, instance.language)
                        Logger.warning(
                            f"Recorded instance {instance.corpus_combination_id} in "
                            f"{lang_key}/{BaseStorage.SKIPPED_NAME}."
                        )
                    except StorageError as err:
                        Logger.error(
                            f"Failed to record skipped instance "
                            f"{instance.corpus_combination_id}: {err}"
                        )
                break
            continue

        consecutive_failures = 0
        accepted += 1

        # Always dump accepted conversations under output/<run_id>/ so the JSON
        # (and a metadata sidecar) is available locally regardless of MODE.
        try:
            local_dir = save_local_output(instance, index, result, models=models)
            Logger.success(f"Wrote local output → {local_dir}")
        except OSError as err:
            Logger.warning(f"Failed to write local output for conversation {index}: {err}")

        if storage is not None and checkpoint is not None:
            # Upload the conversation, THEN advance + persist the checkpoint. The
            # conversation lands in the bucket before the checkpoint claims it,
            # so a crash between the two just regenerates it — the checkpoint
            # never points at a missing file.
            metadata_text = _build_metadata_text(
                f"instance_{instance.corpus_combination_id:04d}_conv{index:04d}",
                instance,
                index,
                result,
            )
            transcript_text = result.get("transcript") or ""
            try:
                lock = storage_lock or threading.Lock()
                with lock:
                    path = storage.save_conversation(
                        instance.corpus_combination_id,
                        index,
                        _conversation_payload(instance, index, result, models=models),
                        language=instance.language,
                        metadata_text=metadata_text,
                        transcript_text=transcript_text or None,
                    )
                    checkpoint.record(progress, duration)
                    storage.save_checkpoint(checkpoint, instance.language)
                    if corpus_budget is not None:
                        corpus_budget.add(duration)
            except StorageError as err:
                Logger.error(f"Storage failure on conversation {index}, aborting instance: {err}")
                break
            duration_bar.update(duration)
            duration_bar.set_postfix_str(
                f"conv {index}, accepted {accepted}, +{duration:.0f}s",
                refresh=True,
            )
            Logger.success(
                f"Saved {path} (+{duration:.0f}s) — "
                f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                bold=True,
            )
        else:
            # Development: count locally, don't upload to the remote bucket.
            progress.generated_sec += duration
            progress.conversation_count += 1
            if corpus_budget is not None:
                corpus_budget.add(duration)
            duration_bar.update(duration)
            duration_bar.set_postfix_str(
                f"conv {index}, accepted {accepted}, +{duration:.0f}s",
                refresh=True,
            )
            Logger.success(
                f"Conversation {index} accepted (+{duration:.0f}s) — "
                f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                bold=True,
            )

    duration_bar.close()
    Logger.success(
        f"Instance {instance.corpus_combination_id} complete: "
        f"{progress.generated_sec:.0f}s across {progress.conversation_count} conversation(s).",
        bold=True,
    )
    return progress.generated_sec


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, 11),
        metavar="N",
        help="Number of parallel instance workers (1–10). Default: 1.",
    )
    parser.add_argument(
        "--corpus-size",
        type=float,
        default=None,
        metavar="MINUTES",
        help=(
            "Stop once total accepted conversation duration for the filtered "
            "language(s) reaches this many minutes."
        ),
    )
    parser.add_argument(
        "--tokenstats",
        action="store_true",
        help=(
            "Print input/output token usage and cost summary from conversations "
            "stored in the HuggingFace bucket, then exit."
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="With --tokenstats, re-download all conversation JSON from HF (ignore local cache).",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class RunnerHandle:
    """Runner plus the generation/validation providers used for this invocation."""

    runner: ConversationRunner
    generation_provider: str
    generation_model: str
    validation_provider: str
    validation_model: str

    @property
    def models(self) -> ModelInfo:
        return ModelInfo(
            generation_provider=self.generation_provider,
            generation_model=self.generation_model,
            validation_provider=self.validation_provider,
            validation_model=self.validation_model,
        )


def _get_runner(
    runner_cache: dict[tuple[LLMProvider, LLMProvider], RunnerHandle],
    model: str | None,
    validation: str | None,
    language: str | None,
) -> RunnerHandle:
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
        runner_cache[cache_key] = RunnerHandle(
            runner=ConversationRunner(
                llm=generation_llm,
                validation_llm=validation_llm,
                max_generation_attempts=3,
                max_format_attempts=3,
            ),
            generation_provider=generation_provider.value,
            generation_model=generation_llm.model,
            validation_provider=validation_provider.value,
            validation_model=validation_llm.model,
        )
    return runner_cache[cache_key]


_thread_local = threading.local()


def _thread_runner_cache() -> dict[tuple[LLMProvider, LLMProvider], RunnerHandle]:
    cache = getattr(_thread_local, "runner_cache", None)
    if cache is None:
        cache = {}
        _thread_local.runner_cache = cache
    return cache


def _languages_in_scope(corpus_df: Any, language: str | None) -> list[str]:
    if language:
        return [BaseStorage.normalize_language(language)]
    langs = corpus_df["language"].str.lower().unique().tolist()
    return sorted(BaseStorage.normalize_language(lang) for lang in langs)


def _build_work_indices(
    corpus_df: Any,
    checkpoint_caches: dict[str, Checkpoint] | None,
    skipped_caches: dict[str, SkippedRegistry] | None,
) -> tuple[list[int], dict[str, int]]:
    """Pick corpus rows to run, highest ``joint_probability`` first.

    Skips instances already completed (checkpoint) or abandoned (skipped.json)
    so production runs resume without walking thousands of finished rows.
    """
    stats = {"skipped_complete": 0, "skipped_abandoned": 0, "pending": 0}
    candidates: list[tuple[int, float]] = []

    for i in range(len(corpus_df)):
        row = corpus_df.iloc[i]
        instance_id = int(row["corpus_combination_id"])
        lang = BaseStorage.normalize_language(str(row["language"]))
        target_sec = float(row.get("duration_sec") or 0)

        if skipped_caches and lang in skipped_caches and instance_id in skipped_caches[lang]:
            stats["skipped_abandoned"] += 1
            continue

        if checkpoint_caches and lang in checkpoint_caches:
            prog = checkpoint_caches[lang].instances.get(str(instance_id))
            if prog and target_sec > 0 and prog.generated_sec >= target_sec:
                stats["skipped_complete"] += 1
                continue

        joint_prob = float(row.get("joint_probability") or 0)
        candidates.append((i, joint_prob))
        stats["pending"] += 1

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [i for i, _ in candidates], stats


def _corpus_sec_from_checkpoints(
    corpus_df: Any,
    checkpoint_caches: dict[str, Checkpoint],
    *,
    language: str | None = None,
) -> float:
    """Sum accepted duration already recorded for instances in this run scope."""
    ids = set(corpus_df["corpus_combination_id"].tolist())
    total = 0.0
    for lang, checkpoint in checkpoint_caches.items():
        if language and lang != BaseStorage.normalize_language(language):
            continue
        for prog in checkpoint.instances.values():
            if prog.corpus_combination_id in ids:
                total += prog.generated_sec
    return total


def _process_corpus_index(
    index: int,
    corpus_df: Any,
    args: argparse.Namespace,
    *,
    storage: BaseStorage | None,
    checkpoint_caches: dict[str, Checkpoint] | None,
    skipped_caches: dict[str, SkippedRegistry] | None,
    max_conversations: int | None,
    corpus_budget: CorpusBudget | None,
    storage_lock: threading.Lock | None,
) -> float:
    """Process one corpus row (used by sequential and parallel drivers)."""
    if corpus_budget is not None and not corpus_budget.can_continue():
        return 0.0

    row: dict[str, Any] = {str(k): v for k, v in corpus_df.iloc[index].to_dict().items()}
    instance = CorpusInstance.from_dict(row)
    lang_key = BaseStorage.normalize_language(instance.language)

    skipped: SkippedRegistry | None = None
    if storage is not None and skipped_caches is not None:
        lock = storage_lock or threading.Lock()
        with lock:
            if lang_key not in skipped_caches:
                skipped_caches[lang_key] = storage.load_skipped(instance.language)
            skipped = skipped_caches[lang_key]

    if skipped is not None and instance.corpus_combination_id in skipped:
        Logger.warning(
            f"Skipping instance {instance.corpus_combination_id} — "
            f"already recorded in {lang_key}/{BaseStorage.SKIPPED_NAME}."
        )
        return 0.0

    runner = _get_runner(
        _thread_runner_cache(),
        args.model,
        args.validation_model,
        instance.language,
    )
    return process_instance(
        runner.runner,
        instance,
        storage,
        checkpoint_caches,
        skipped_caches,
        max_conversations,
        corpus_budget,
        storage_lock,
        models=runner.models,
    )


def main(argv: list[str] | None = None) -> None:
    # Load API keys / settings from conversations_generator/config.json and
    # mirror them into os.environ for any third-party SDK that only reads env.
    apply_to_environ()
    args = _parse_args(argv)

    if args.tokenstats:
        Logger.step(
            "Scanning HuggingFace bucket metadata (metadata.txt — not conversation.json)..."
        )
        print_token_stats(language=args.language, refresh=args.refresh_cache)
        return

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

    if args.language:
        mask = corpus_df["language"].str.lower() == args.language
        corpus_df = corpus_df[mask].reset_index(drop=True)
        if corpus_df.empty:
            Logger.error(f"No corpus instances found for language={args.language!r}.")
            return
        Logger.step(
            f"Filtered corpus to language={args.language!r}: {len(corpus_df)} instance(s)."
        )

    hindi_note = "" if args.model else " (Hindi defaults to sarvam when --model is omitted)"
    Logger.info(
        f"Providers — generation(--model)={args.model or DEFAULT_GENERATION_PROVIDER.value}"
        f"{hindi_note}, validation/formatter(--validation-model)={args.validation_model}"
    )
    Logger.info(f"Parallel workers: {args.workers}")

    # Storage is production-only. Checkpoints/skipped are loaded per language.
    storage: BaseStorage | None = None
    checkpoint_caches: dict[str, Checkpoint] | None = None
    skipped_caches: dict[str, SkippedRegistry] | None = None
    # In development, cap at a single conversation per instance; production
    # chases each instance's full duration target (unless --corpus-size is set).
    max_conversations: int | None = None
    if production:
        Logger.step(
            f"PRODUCTION run — processing corpus instances "
            f"(bucket layout: <language>/instance_<id>/conversation_<n>/)."
        )
        storage = HuggingFaceStorage()
        scope_languages = _languages_in_scope(corpus_df, args.language)
        Logger.step(
            f"Syncing checkpoints for {', '.join(scope_languages)} "
            f"(cached under .cache/hf_control/ after first run)..."
        )
        checkpoint_caches, skipped_caches = storage.preload_language_state(scope_languages)
        work_stats: dict[str, int] = {}
        indices, work_stats = _build_work_indices(corpus_df, checkpoint_caches, skipped_caches)
        Logger.info(
            f"Instance queue: {work_stats['pending']} pending, "
            f"{work_stats['skipped_complete']} already complete, "
            f"{work_stats['skipped_abandoned']} abandoned — "
            f"ordered by joint_probability (distribution priority)."
        )
    else:
        Logger.step("DEVELOPMENT run — generating conversations (no HF upload).")
        indices = [
            234, 0, 135, 59, 3278, 6599, 3235, 5, 4921, 4751, 2225, 2882, 2563,
        ]
        max_conversations = 1

    corpus_target_sec = args.corpus_size * 60.0 if args.corpus_size else None
    initial_sec = (
        _corpus_sec_from_checkpoints(
            corpus_df, checkpoint_caches, language=args.language
        )
        if corpus_target_sec and checkpoint_caches
        else 0.0
    )
    corpus_budget: CorpusBudget | None = None
    if corpus_target_sec is not None:
        corpus_budget = CorpusBudget(corpus_target_sec, initial_sec)
        Logger.step(
            f"Corpus size cap: {args.corpus_size} min ({corpus_target_sec:.0f}s) — "
            f"already generated {initial_sec / 60:.2f} min from checkpoint."
        )
        if production or args.corpus_size:
            max_conversations = None

    # Drop indices outside the filtered corpus (dev smoke list vs language filter).
    indices = [i for i in indices if 0 <= i < len(corpus_df)]
    if not indices:
        Logger.error("No corpus indices to process.")
        return

    storage_lock = threading.Lock() if args.workers > 1 else None
    if args.workers > 1:
        tqdm.set_lock(storage_lock)
    worker_kwargs = dict(
        corpus_df=corpus_df,
        args=args,
        storage=storage,
        checkpoint_caches=checkpoint_caches,
        skipped_caches=skipped_caches,
        max_conversations=max_conversations,
        corpus_budget=corpus_budget,
        storage_lock=storage_lock,
    )

    if args.workers <= 1:
        instances_bar = tqdm(
            indices,
            desc="Corpus instances",
            unit="inst",
            dynamic_ncols=True,
            bar_format=_INSTANCE_BAR_FMT,
        )
        for i in instances_bar:
            if corpus_budget is not None and not corpus_budget.can_continue():
                Logger.info("Corpus size target reached; stopping.")
                break
            _process_corpus_index(i, **worker_kwargs)
            postfix = _corpus_postfix(corpus_budget, args.corpus_size)
            if postfix:
                instances_bar.set_postfix_str(postfix, refresh=True)
        instances_bar.close()
    else:
        Logger.step(f"Running {len(indices)} instance(s) with {args.workers} worker(s).")
        instances_bar = tqdm(
            total=len(indices),
            desc="Corpus instances",
            unit="inst",
            dynamic_ncols=True,
            bar_format=_INSTANCE_BAR_FMT,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_corpus_index, i, **worker_kwargs) for i in indices]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as err:  # noqa: BLE001
                    Logger.error(f"Worker failed: {err}")
                instances_bar.update(1)
                postfix = _corpus_postfix(corpus_budget, args.corpus_size)
                if postfix:
                    instances_bar.set_postfix_str(postfix, refresh=True)
                if corpus_budget is not None and not corpus_budget.can_continue():
                    Logger.info("Corpus size target reached; draining workers.")
                    break
        instances_bar.close()

    if corpus_budget is not None:
        Logger.success(
            f"Corpus progress: {corpus_budget.generated_sec / 60:.2f} / "
            f"{args.corpus_size} min accepted.",
            bold=True,
        )

    Logger.divider()
    Logger.success("All requested instances processed.", bold=True)


if __name__ == "__main__":
    main()