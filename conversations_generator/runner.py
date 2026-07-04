"""Orchestration for the conversation-generation pipeline.

A single :class:`BaseLLM` is created once and shared by every agent, so switching
provider (Groq by default) is one argument. The runner drives the pipeline in
stages — topic generation, conversation generation (a plain-text *generator*
followed by a JSON *formatter* that assigns deterministic timing/overlap
metadata), manual timing validation, and LLM agent validation — so every
generated conversation is mechanically checked (overlap/interruption/backchannel
timing, duration, turn-type distribution) before it's handed back to the caller.

    python -m conversations_generator.runner
    python -m conversations_generator.runner --language=hindi
    python -m conversations_generator.runner --language=english --model=gemini
    python -m conversations_generator.runner --model=sarvam --validation-model=gemma

``--language`` restricts the run to one corpus language (``hindi`` / ``hinglish``
/ ``english``); omit it to process every language, as before.

``--model`` is **generation only** (topic + conversation transcript) and always
wins when given, for any language. Only when it's *omitted* does Hindi default
to Sarvam and other languages default to Krutrim.

``--validation-model`` is shared by the **formatter** and the **LLM agent validator**
(not the deterministic manual checks). Defaults to ``gemma``.
"""

from __future__ import annotations

import argparse
import json
import random
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
from .agents.conversation_validator_agent import ConversationValidatorAgent, AgentValidationReport
from .agents.conversation_validator_manual import ConversationValidatorManual, ValidationReport
from .configuration_reader import (
    apply_to_environ,
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
    create_llm,
    resolve_provider,
)
from .loaders import read_corpus_instances
from .logger import Logger
from .models import CorpusInstance
from .storage import BaseStorage, Checkpoint, HuggingFaceStorage, InstanceProgress, StorageError

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


class ConversationRunner:
    """Drives the pipeline agents stage by stage.

    Parameters
    ----------
    llm : BaseLLM | None
        Generation LLM — topic agent + conversation generator only.
        ``None`` lets those agents fall back to their own default (Groq).
    validation_llm : BaseLLM | None
        Shared by the formatter agent and the LLM agent validator. When
        ``None``, falls back to ``llm`` (same provider for everything).
    validator : ConversationValidatorManual | None
        Deterministic timing/overlap validator run after generation. Defaults
        to ``ConversationValidatorManual()`` with its standard thresholds.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        validation_llm: BaseLLM | None = None,
        validator: ConversationValidatorManual | None = None,
        max_agent_attempts: int = 3,
        max_manual_attempts: int = 3,
        max_agent_validation_retries: int = 3,
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
        self.agent_validator = ConversationValidatorAgent(self.validation_llm)
        # Repairs a failing conversation with targeted per-turn edits instead of
        # regenerating the whole thing. Uses the generation LLM (it owns dialogue
        # content), falling back to the validation LLM if generation is unset.
        self.editor_agent = ConversationEditorAgent(llm if llm is not None else self.validation_llm)
        self.max_agent_attempts = max(1, max_agent_attempts)
        self.max_manual_attempts = max(1, max_manual_attempts)
        # How many times to retry the agent-validation CALL if it errors out
        # (bad/unparseable LLM response) before bypassing the step entirely.
        self.max_agent_validation_retries = max(1, max_agent_validation_retries)
        # How many targeted-edit passes to try before giving up and regenerating.
        self.max_edit_attempts = max(1, max_edit_attempts)

    # ------------------------------------------------------------------ #
    # Stage 1: topic
    # ------------------------------------------------------------------ #
    def generate_topic(self, **profile: Any) -> dict[str, str]:
        """Produce the next single topic (see ``TopicGeneratorAgent.run``)."""
        Logger.info(f"Generating topic for {profile.get('language', 'unknown')}...")
        topic = self.topic_agent.run(**profile)
        Logger.success(f"Topic generated: {topic.get('title', 'Unknown Title')}")
        return topic

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
        """Run the full pipeline with a two-level retry loop.

        Flow::

            topic = generate_topic()
            for agent_attempt in range(max_agent_attempts):      # OUTER
                transcript = generator_agent(...)                # one transcript per outer attempt
                for manual_attempt in range(max_manual_attempts):  # INNER
                    turns = formatter_agent(transcript, ...)
                    if manual_validation(turns) passes: break
                    # else feed manual errors + previous formatted turns back
                    # to the FORMATTER (transcript unchanged) and retry
                if manual validation still failing: regenerate transcript (outer)
                if agent_validation(turns) passes: done
                # else FIRST try targeted edits (Stage 4b): patch only the flagged
                # turns in place, re-lay-out timing, re-validate — up to
                # max_edit_attempts passes. If that fixes it: done, no regeneration.
                # Only if editing can't be applied / doesn't pass do we fall back to
                # feeding agent errors + previous TRANSCRIPT back to the GENERATOR
                # and regenerating the whole conversation (outer).

        So the **inner** loop only re-runs the formatter to fix *formatting*, the
        **edit** stage surgically fixes flagged *content* without discarding the
        good turns, and the **outer** loop re-runs the generator only when editing
        isn't enough — each stage gets feedback targeted at the artefact it owns.

        Returns
        -------
        dict with keys: topic, transcript, turns, manual_validation,
            agent_validation, agent_validation_bypassed, passed.
        """
        Logger.step(f"Starting pipeline for language: {profile.get('language', 'unknown')}")

        # Decide ONCE per conversation whether this one is number-rich, drawn from
        # NUMBER_INCLUSION_PERCENTAGE (default 50%). Fixed for the whole pipeline
        # (topic + every regeneration/edit) so the topic and dialogue stay aligned.
        include_numbers = random.random() < get_number_inclusion_percentage()
        Logger.info(
            f"Number inclusion: {'ON — numbers + reasoning' if include_numbers else 'OFF — qualitative'}"
        )

        topic = self.generate_topic(include_numbers=include_numbers, **profile)

        # Draw one exact target duration for this conversation from the Gaussian
        # so lengths follow the intended 4–8 min distribution. Sampled once and
        # reused across all retries so every regeneration aims for the same target.
        target_duration_sec = sample_target_duration_sec()
        Logger.info(
            f"Target conversation duration: {target_duration_sec:.1f}s "
            f"({target_duration_sec / 60:.2f} min)"
        )

        turns: list[dict[str, Any]] = []
        transcript: str = ""
        manual_report: ValidationReport | None = None
        agent_report: AgentValidationReport | None = None
        agent_validation_bypassed = False

        # Feedback threaded to the GENERATOR across outer attempts.
        generator_feedback: str | None = None
        previous_transcript: str | None = None

        # Attempt counters for local metadata (how many tries it took to pass).
        agent_attempts_used = 0
        manual_attempts_used = 0
        agent_validation_attempts_used = 0

        for agent_attempt in range(1, self.max_agent_attempts + 1):
            agent_attempts_used = agent_attempt
            if agent_attempt > 1:
                Logger.retry(
                    f"Regenerating conversation (agent attempt {agent_attempt}/{self.max_agent_attempts})"
                )
            else:
                Logger.step("Stage 2: Conversation Generation")

            # ---- Stage 2a: generate the plain-text transcript ONCE per outer attempt ----
            Logger.info("Generating conversation transcript (plain text)...")
            try:
                transcript = self.generator_agent.run(
                    title=topic["title"],
                    context=topic.get("context", ""),
                    conversation_type=topic.get("conversation_type"),
                    previous_transcript=previous_transcript,
                    feedback=generator_feedback,
                    target_duration_sec=target_duration_sec,
                    include_numbers=include_numbers,
                    **profile,
                )
            except (ValueError, LLMError) as err:
                Logger.warning(f"Transcript generation failed: {err}")
                manual_report = None
                generator_feedback = f"Conversation generation error on the previous attempt: {err}"
                previous_transcript = None
                continue

            # ---- Stage 2b + 3: format + manual validation (INNER loop, formatter only) ----
            Logger.step("Stage 3: Formatting & Manual Validation")
            manual_report = None
            turns = []
            formatter_feedback: str | None = None
            previous_output: list[dict[str, Any]] | None = None
            manual_attempts_used = 0
            for manual_attempt in range(1, self.max_manual_attempts + 1):
                manual_attempts_used = manual_attempt
                if manual_attempt > 1:
                    Logger.retry(
                        f"Re-formatting (manual attempt {manual_attempt}/{self.max_manual_attempts})"
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
                    )
                except (ValueError, LLMError) as err:
                    Logger.warning(f"Formatting failed: {err}")
                    manual_report = None
                    formatter_feedback = f"Formatting error on the previous attempt: {err}"
                    previous_output = None
                    continue

                Logger.info("Running deterministic manual validation...")
                manual_report = self.validate_conversation(turns)
                if not manual_report.has_errors:
                    Logger.success(f"Manual validation passed on attempt {manual_attempt}!")
                    break

                Logger.warning(f"Manual validation failed with {len(manual_report.errors)} errors.")
                # Feedback for the next FORMATTER retry (inner loop only). The
                # transcript stays the same; we give the formatter its own errors
                # plus the output it produced so it can fix the formatting.
                formatter_feedback = "Manual Validation Errors:\n" + "\n".join(
                    f"- [Turn {e.turn_id}]: {e.message}" for e in manual_report.errors
                )
                previous_output = turns or None

            # If formatting couldn't pass manual validation, the transcript itself
            # is suspect — regenerate it on the next outer attempt.
            if manual_report is None or manual_report.has_errors:
                Logger.error(
                    "Manual validation did not pass after all formatter retries; "
                    "regenerating the conversation."
                )
                generator_feedback = formatter_feedback or "The formatted conversation failed manual validation."
                previous_transcript = transcript or None
                continue

            # ---- Stage 4: LLM agent (content/realism) validation ----
            Logger.step(f"Stage 4: LLM Agent Validation (agent attempt {agent_attempt}/{self.max_agent_attempts})")
            agent_report, agent_validation_attempts_used = self._run_agent_validation(turns, topic, profile)

            # Still couldn't get a verdict after all retries — bypass the step and
            # accept the conversation, which already passed manual validation.
            if agent_report is None:
                Logger.warning(
                    f"Agent validation unavailable after {self.max_agent_validation_retries} "
                    "retries. Bypassing agent validation for this conversation."
                )
                agent_validation_bypassed = True
                break

            if agent_report.passed:
                Logger.success(
                    f"Agent validation passed! (Realism: {agent_report.realism_score}, "
                    f"Match: {agent_report.corpus_match_score})",
                    bold=True,
                )
                break

            Logger.warning(f"Agent validation failed. Verdict: {agent_report.verdict}")

            # ---- Stage 4b: TARGETED EDITS before falling back to full regeneration ----
            # Try to fix only the flagged turns (edit in place) rather than throwing
            # away the whole conversation and regenerating from scratch.
            edit_result = self._repair_by_editing(turns, topic, agent_report, profile)
            if edit_result is not None:
                turns, edited_manual_report, edited_agent_report = edit_result
                manual_report = edited_manual_report
                if edited_agent_report is None:
                    # Re-validation was unavailable after edits; accept the edited,
                    # manual-valid conversation (same policy as the bypass above).
                    agent_validation_bypassed = True
                    agent_report = None
                    break
                agent_report = edited_agent_report
                if agent_report.passed:
                    Logger.success(
                        f"Targeted edits fixed the conversation! (Realism: "
                        f"{agent_report.realism_score}, Match: {agent_report.corpus_match_score})",
                        bold=True,
                    )
                    break
                # Edits helped but didn't fully pass — regenerate, now with the
                # LATEST issues from the edited conversation.

            # Feedback for the next GENERATOR attempt (outer loop): pass the agent's
            # errors and the previous TRANSCRIPT (not the formatted turns), since the
            # generator owns the dialogue content the agent validator judged.
            generator_feedback = "Agent Validation Feedback:\n"
            if agent_report and agent_report.feedback:
                generator_feedback += f"{agent_report.feedback}\n"
            if agent_report and agent_report.issues:
                generator_feedback += "\n".join(
                    f"- ({i.severity}) [Turn {i.turn_id}]: {i.description}" for i in agent_report.issues
                )
            previous_transcript = transcript or None

        return {
            "topic": topic,
            "transcript": transcript,
            "turns": turns,
            "manual_validation": manual_report,
            "agent_validation": agent_report,
            "agent_validation_bypassed": agent_validation_bypassed,
            "target_duration_sec": target_duration_sec,
            "include_numbers": include_numbers,
            "agent_attempts_used": agent_attempts_used,
            "manual_attempts_used": manual_attempts_used,
            "agent_validation_attempts_used": agent_validation_attempts_used,
            "max_agent_attempts": self.max_agent_attempts,
            "max_manual_attempts": self.max_manual_attempts,
            "max_agent_validation_retries": self.max_agent_validation_retries,
            # Accept the conversation if manual validation passed AND agent
            # validation either passed or was bypassed after exhausting retries.
            "passed": bool(
                manual_report
                and not manual_report.has_errors
                and (agent_validation_bypassed or (agent_report and agent_report.passed))
            ),
        }

    # ------------------------------------------------------------------ #
    # Stage 4 helpers
    # ------------------------------------------------------------------ #
    def _run_agent_validation(
        self,
        turns: list[dict[str, Any]],
        topic: dict[str, str],
        profile: dict[str, Any],
    ) -> tuple[AgentValidationReport | None, int]:
        """Call the LLM agent validator, retrying transient call failures.

        Returns ``(report, attempts_used)``; ``report`` is ``None`` if every
        attempt raised (unparseable/failed response), so the caller can bypass.
        """
        report: AgentValidationReport | None = None
        attempts_used = 0
        for validation_attempt in range(1, self.max_agent_validation_retries + 1):
            attempts_used = validation_attempt
            try:
                report = self.agent_validator.run(turns=turns, topic=topic, **profile)
                break
            except (ValueError, LLMError) as err:
                Logger.warning(
                    f"Agent validation call failed "
                    f"(attempt {validation_attempt}/{self.max_agent_validation_retries}): {err}"
                )
        return report, attempts_used

    def _repair_by_editing(
        self,
        turns: list[dict[str, Any]],
        topic: dict[str, str],
        agent_report: AgentValidationReport,
        profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], ValidationReport, AgentValidationReport | None] | None:
        """Fix a failing conversation with targeted per-turn edits.

        Repeatedly asks the editor agent for a minimal patch addressing the
        validator's issues, applies it (which deterministically re-lays-out
        timing), re-checks manual validation, then re-judges with the agent
        validator — up to ``max_edit_attempts`` passes.

        Returns ``(edited_turns, manual_report, agent_report)`` where
        ``agent_report`` is the latest verdict (or ``None`` if re-validation was
        unavailable). Returns ``None`` when editing can't be applied at all
        (editor error, no edits proposed, patch rejected, or the edited turns
        fail manual validation) — the caller then falls back to full regeneration.
        """
        current_turns = turns
        current_report = agent_report
        manual_report: ValidationReport | None = None

        for edit_attempt in range(1, self.max_edit_attempts + 1):
            Logger.step(
                f"Stage 4b: Targeted edit repair (attempt {edit_attempt}/{self.max_edit_attempts})"
            )
            try:
                edits = self.editor_agent.run(
                    turns=current_turns,
                    issues=current_report.issues,
                    feedback=current_report.feedback,
                    **profile,
                )
            except (ValueError, LLMError) as err:
                Logger.warning(f"Editor call failed: {err}. Falling back to regeneration.")
                return None

            if not edits:
                Logger.info("Editor proposed no edits. Falling back to regeneration.")
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
                Logger.warning(f"Applying edits failed: {err}. Falling back to regeneration.")
                return None

            manual_report = self.validate_conversation(edited_turns)
            if manual_report.has_errors:
                Logger.warning(
                    f"Edited conversation failed manual validation "
                    f"({len(manual_report.errors)} errors). Falling back to regeneration."
                )
                return None

            Logger.success(f"Edits applied and passed manual validation (attempt {edit_attempt}).")
            current_turns = edited_turns

            # Re-judge the edited conversation.
            new_report, _ = self._run_agent_validation(current_turns, topic, profile)
            if new_report is None:
                Logger.warning("Agent re-validation unavailable after edits; accepting edited conversation.")
                return current_turns, manual_report, None
            if new_report.passed:
                return current_turns, manual_report, new_report

            Logger.warning(
                f"Edited conversation still failing (verdict {new_report.verdict}); "
                "trying another edit pass." if edit_attempt < self.max_edit_attempts
                else f"Edited conversation still failing (verdict {new_report.verdict})."
            )
            current_report = new_report

        # Edit passes exhausted without a PASS — hand back the best-effort edited
        # turns and the latest verdict so the caller decides (regenerate).
        assert manual_report is not None
        return current_turns, manual_report, current_report


# Safety cap: give up on an instance after this many *consecutive* generations
# that fail validation, so a persistently-failing profile can't loop forever.
MAX_CONSECUTIVE_FAILURES = 10

# Local on-disk dump of every accepted conversation (dev and prod).
# Layout: <repo>/output/<run_id>/conversation.json + metadata.txt
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


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
    agent_report = result.get("agent_validation")
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

    agent_attempts = result.get("agent_attempts_used", 0)
    manual_attempts = result.get("manual_attempts_used", 0)
    validation_attempts = result.get("agent_validation_attempts_used", 0)
    lines += [
        "",
        "## Retries",
        f"agent_attempts_used: {agent_attempts} / {result.get('max_agent_attempts')}",
        f"manual_attempts_used: {manual_attempts} / {result.get('max_manual_attempts')}",
        f"agent_validation_attempts_used: {validation_attempts} / {result.get('max_agent_validation_retries')}",
        f"agent_retries: {max(0, int(agent_attempts) - 1)}",
        f"manual_retries: {max(0, int(manual_attempts) - 1)}",
        f"agent_validation_retries: {max(0, int(validation_attempts) - 1)}",
        f"agent_validation_bypassed: {result.get('agent_validation_bypassed', False)}",
        "",
        "## Agent validation",
    ]
    if agent_report is not None:
        lines += [
            f"verdict: {getattr(agent_report, 'verdict', '')}",
            f"realism_score: {getattr(agent_report, 'realism_score', '')}",
            f"corpus_match_score: {getattr(agent_report, 'corpus_match_score', '')}",
        ]
        field_matches = getattr(agent_report, "corpus_field_matches", None) or {}
        if field_matches:
            lines.append("corpus_field_matches:")
            for key, matched in field_matches.items():
                lines.append(f"  {key}: {matched}")
        feedback = getattr(agent_report, "feedback", "") or ""
        if feedback:
            lines += ["", "feedback:", feedback]
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


def process_instance(
    runner: ConversationRunner,
    instance: CorpusInstance,
    storage: BaseStorage | None = None,
    checkpoint: Checkpoint | None = None,
    max_conversations: int | None = None,
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
    """
    target_sec = instance.duration_sec or 0.0

    # In dev (no storage) progress lives only in this local record; in prod it's
    # the shared, resumable checkpoint entry for this instance.
    if storage is not None and checkpoint is not None:
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

    while progress.generated_sec < target_sec:
        if max_conversations is not None and accepted >= max_conversations:
            break
        index += 1
        Logger.divider()
        Logger.info(
            f"Instance {instance.corpus_combination_id} — conversation {index} "
            f"(progress {progress.generated_sec:.0f}/{target_sec:.0f}s, "
            f"{progress.generated_sec / target_sec * 100 if target_sec else 0:.1f}%)"
        )

        result = runner.run(**instance.to_profile())
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
                break
            continue

        consecutive_failures = 0
        accepted += 1

        # Always dump accepted conversations under output/<run_id>/ so the JSON
        # (and a metadata sidecar) is available locally regardless of MODE.
        try:
            local_dir = save_local_output(instance, index, result)
            Logger.success(f"Wrote local output → {local_dir}")
        except OSError as err:
            Logger.warning(f"Failed to write local output for conversation {index}: {err}")

        if storage is not None and checkpoint is not None:
            # Upload the conversation, THEN advance + persist the checkpoint. The
            # conversation lands in the bucket before the checkpoint claims it,
            # so a crash between the two just regenerates it — the checkpoint
            # never points at a missing file.
            try:
                path = storage.save_conversation(
                    instance.corpus_combination_id,
                    index,
                    _conversation_payload(instance, index, result),
                )
                checkpoint.record(progress, duration)
                storage.save_checkpoint(checkpoint)
            except StorageError as err:
                Logger.error(f"Storage failure on conversation {index}, aborting instance: {err}")
                break
            Logger.success(
                f"Saved {path} (+{duration:.0f}s) — "
                f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                bold=True,
            )
        else:
            # Development: count locally, don't upload to the remote bucket.
            progress.generated_sec += duration
            progress.conversation_count += 1
            Logger.success(
                f"Conversation {index} accepted (+{duration:.0f}s) — "
                f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                bold=True,
            )

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
            max_agent_attempts=3,
            max_manual_attempts=3,
        )
    return runner_cache[cache_key]


def main(argv: list[str] | None = None) -> None:
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

    # Runners are cached by (generation_provider, validation_provider) because
    # generation can differ per instance (Hindi → Sarvam) while validation is
    # fixed for the whole CLI invocation.
    runner_cache: dict[tuple[LLMProvider, LLMProvider], ConversationRunner] = {}

    # Storage + checkpoint are production-only. Dev runs entirely in memory.
    storage: BaseStorage | None = None
    checkpoint: Checkpoint | None = None
    # In development, cap at a single conversation per instance; production
    # chases each instance's full duration target.
    max_conversations: int | None = None
    if production:
        Logger.step(f"PRODUCTION run — processing all {len(corpus_df)} instances in sequence.")
        storage = HuggingFaceStorage()
        checkpoint = storage.load_checkpoint()
        Logger.info(f"Loaded checkpoint with {len(checkpoint.instances)} instance record(s).")
        indices = range(len(corpus_df))
    else:
        Logger.step("DEVELOPMENT run — generating a single conversation per instance (no upload).")
        # Without a language filter, keep the original hand-picked indices for
        # variety across the full corpus; with a filter, just take the first
        # few rows of the (already language-scoped) corpus.
        indices = [i for i in (234, 0, 135) if i < len(corpus_df)] if not args.language else list(
            range(min(3, len(corpus_df)))
        )
        max_conversations = 1

    for i in indices:
        row: dict[str, Any] = {str(k): v for k, v in corpus_df.iloc[i].to_dict().items()}
        instance = CorpusInstance.from_dict(row)
        runner = _get_runner(runner_cache, args.model, args.validation_model, instance.language)
        process_instance(runner, instance, storage, checkpoint, max_conversations)

    Logger.divider()
    Logger.success("All requested instances processed.", bold=True)


if __name__ == "__main__":
    main()