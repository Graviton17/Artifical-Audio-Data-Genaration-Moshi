"""Orchestration for the conversation-generation pipeline.

A single :class:`BaseLLM` is created once and shared by every agent, so switching
provider (Groq by default) is one argument. The runner drives the pipeline in
stages — topic generation, conversation generation (a plain-text *generator*
followed by a JSON *formatter* that assigns deterministic timing/overlap
metadata), manual timing validation, and LLM agent validation — so every
generated conversation is mechanically checked (overlap/interruption/backchannel
timing, duration, turn-type distribution) before it's handed back to the caller.

    python -m conversations_generator.runner
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from . import settings
from .agents import (
    ConversationEditorAgent,
    ConversationFormatterAgent,
    ConversationGeneratorAgent,
    TopicGeneratorAgent,
)
from .agents.conversation_validator_agent import ConversationValidatorAgent, AgentValidationReport
from .agents.conversation_validator_manual import ConversationValidatorManual, ValidationReport
from .llm import BaseLLM, KrutrimLLM, LLMError
from .loaders import load_env, read_corpus_instances
from .logger import Logger
from .models import CorpusInstance
from .storage import BaseStorage, Checkpoint, HuggingFaceStorage, InstanceProgress, StorageError


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
    """Drives the pipeline agents over a shared LLM, stage by stage.

    Parameters
    ----------
    llm : BaseLLM | None
        Shared LLM instance passed to every agent. ``None`` lets each agent
        fall back to its own default (Groq).
    validator : ConversationValidatorManual | None
        Deterministic timing/overlap validator run after generation. Defaults
        to ``ConversationValidatorManual()`` with its standard thresholds.
    max_generation_attempts : int
        How many times to regenerate the conversation if manual validation
        reports errors (not just warnings) before giving up and returning the
        last attempt anyway. Defaults to 1 (no retry).
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        validator: ConversationValidatorManual | None = None,
        max_agent_attempts: int = 3,
        max_manual_attempts: int = 3,
        max_agent_validation_retries: int = 3,
        max_edit_attempts: int = 2,
    ) -> None:
        self.llm = llm
        self.topic_agent = TopicGeneratorAgent(llm)
        # Conversation generation is a two-stage pipeline: the generator writes
        # tagged plain text, the formatter turns it into schema JSON with
        # deterministic timing/overlap metadata.
        self.generator_agent = ConversationGeneratorAgent(llm)
        self.formatter_agent = ConversationFormatterAgent(llm)
        self.validator = validator or ConversationValidatorManual()
        self.agent_validator = ConversationValidatorAgent(llm)
        # Repairs a failing conversation with targeted per-turn edits instead of
        # regenerating the whole thing (see _repair_by_editing).
        self.editor_agent = ConversationEditorAgent(llm)
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
                # else feed agent errors + previous TRANSCRIPT (not the formatted
                # turns) back to the GENERATOR and regenerate (outer)

        So the **inner** loop only re-runs the formatter to fix *formatting*, while
        the **outer** loop re-runs the generator to fix *content* — each stage gets
        feedback targeted at the artefact it owns.

        Returns
        -------
        dict with keys: topic, transcript, turns, manual_validation,
            agent_validation, agent_validation_bypassed, passed.
        """
        Logger.step(f"Starting pipeline for language: {profile.get('language', 'unknown')}")

        # Decide ONCE per conversation whether this one is number-rich, drawn from
        # NUMBER_INCLUSION_PERCENTAGE. Fixed for the whole pipeline (topic + every
        # regeneration/edit) so the topic and dialogue stay aligned.
        include_numbers = random.random() < settings.get_number_inclusion_percentage()
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

        for agent_attempt in range(1, self.max_agent_attempts + 1):
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
            for manual_attempt in range(1, self.max_manual_attempts + 1):
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
            agent_report = self._run_agent_validation(turns, topic, profile)

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
                turns, manual_report, edited_agent_report = edit_result
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
    ) -> AgentValidationReport | None:
        """Call the LLM agent validator, retrying transient call failures.

        Returns the report, or ``None`` if every attempt raised
        (unparseable/failed response) so the caller can bypass the step.
        """
        for validation_attempt in range(1, self.max_agent_validation_retries + 1):
            try:
                return self.agent_validator.run(turns=turns, topic=topic, **profile)
            except (ValueError, LLMError) as err:
                Logger.warning(
                    f"Agent validation call failed "
                    f"(attempt {validation_attempt}/{self.max_agent_validation_retries}): {err}"
                )
        return None

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
            new_report = self._run_agent_validation(current_turns, topic, profile)
            if new_report is None:
                Logger.warning("Agent re-validation unavailable after edits; accepting edited conversation.")
                return current_turns, manual_report, None
            if new_report.passed:
                return current_turns, manual_report, new_report

            Logger.warning(
                f"Edited conversation still failing (verdict {new_report.verdict}); "
                + ("trying another edit pass." if edit_attempt < self.max_edit_attempts
                   else "edit passes exhausted.")
            )
            current_report = new_report

        # Edit passes exhausted without a PASS — hand back the best-effort edited
        # turns and the latest verdict so the caller decides (regenerate).
        assert manual_report is not None
        return current_turns, manual_report, current_report


# Safety cap: give up on an instance after this many *consecutive* generations
# that fail validation, so a persistently-failing profile can't loop forever.
MAX_CONSECUTIVE_FAILURES = 10


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
        "turns": result.get("turns", []),
    }


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
            # Development: count locally, don't upload.
            progress.generated_sec += duration
            progress.conversation_count += 1
            Logger.success(
                f"Conversation {index} accepted (+{duration:.0f}s, not uploaded in dev) — "
                f"total {progress.generated_sec:.0f}/{target_sec:.0f}s",
                bold=True,
            )

    Logger.success(
        f"Instance {instance.corpus_combination_id} complete: "
        f"{progress.generated_sec:.0f}s across {progress.conversation_count} conversation(s).",
        bold=True,
    )
    return progress.generated_sec


def main() -> None:
    load_env()  # pull GROQ_API_KEY / GEMINI_API_KEY from .env

    # ``ENV=production`` processes every corpus instance in sequence; anything
    # else (the default) is treated as development and runs only the first row.
    is_production = settings.is_production()

    krutrim_llm = KrutrimLLM()
    runner = ConversationRunner(llm=krutrim_llm, max_agent_attempts=3, max_manual_attempts=3)

    corpus_path = Path(__file__).resolve().parent / "data" / "corpus_instances.jsonl"
    corpus_df = read_corpus_instances(str(corpus_path))

    # Storage + checkpoint are production-only. Dev runs entirely in memory.
    storage: BaseStorage | None = None
    checkpoint: Checkpoint | None = None
    # In development, cap at a single conversation for iloc[0]; production chases
    # each instance's full duration target.
    max_conversations: int | None = None
    if is_production:
        Logger.step(f"PRODUCTION run — processing all {len(corpus_df)} instances in sequence.")
        storage = HuggingFaceStorage()
        checkpoint = storage.load_checkpoint()
        Logger.info(f"Loaded checkpoint with {len(checkpoint.instances)} instance record(s).")
        indices = range(len(corpus_df))
    else:
        Logger.step("DEVELOPMENT run — generating a single conversation for instance iloc[0] (no upload).")
        indices = [234, 0, 135]
        max_conversations = 1

    for i in indices:
        row: dict[str, Any] = {str(k): v for k, v in corpus_df.iloc[i].to_dict().items()}
        instance = CorpusInstance.from_dict(row)
        process_instance(runner, instance, storage, checkpoint, max_conversations)

    Logger.divider()
    Logger.success("All requested instances processed.", bold=True)


if __name__ == "__main__":
    main()