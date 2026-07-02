"""Orchestration for the conversation-generation pipeline.

A single :class:`BaseLLM` is created once and shared by every agent, so switching
provider (Groq by default) is one argument. The runner now drives three stages —
topic generation, conversation generation, and manual timing validation — so
every generated conversation is mechanically checked (overlap/interruption/
backchannel timing, duration, turn-type distribution) before it's handed back
to the caller.

    python -m conversations_generator.runner
"""

from __future__ import annotations

import json
import os
import random
import pandas as pd
from pathlib import Path
from typing import Any

from .agents import ConversationGeneratorAgent, TopicGeneratorAgent
from .agents.conversation_validator_agent import ConversationValidatorAgent, AgentValidationReport
from .agents.conversation_validator_manual import ConversationValidatorManual, ValidationReport
from .llm import BaseLLM, GeminiLLM, LLMError
from .logger import Logger
from .models import CorpusInstance
from .storage import BaseStorage, Checkpoint, HuggingFaceStorage, InstanceProgress, StorageError


def load_env(filename: str = ".env") -> None:
    """Load ``KEY=VALUE`` pairs from a .env file into ``os.environ``.

    Walks up from this file to the repo root looking for ``filename``. Existing
    environment variables win, so real env config is never overwritten. Zero
    dependencies; use python-dotenv instead if you later add it.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / filename
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
            return


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
    ) -> None:
        self.llm = llm
        self.topic_agent = TopicGeneratorAgent(llm)
        self.conversation_agent = ConversationGeneratorAgent(llm)
        self.validator = validator or ConversationValidatorManual()
        self.agent_validator = ConversationValidatorAgent(llm)
        self.max_agent_attempts = max(1, max_agent_attempts)
        self.max_manual_attempts = max(1, max_manual_attempts)
        # How many times to retry the agent-validation CALL if it errors out
        # (bad/unparseable LLM response) before bypassing the step entirely.
        self.max_agent_validation_retries = max(1, max_agent_validation_retries)

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
        previous_turns: list[dict[str, Any]] | None = None,
        feedback: str | None = None,
        target_duration_sec: float | None = None,
        **profile: Any,
    ) -> list[dict[str, Any]]:
        """Generate a full conversation from a topic dict.

        Parameters
        ----------
        topic : dict
            Must contain ``title`` and ``context`` (output of stage 1).
        previous_turns : list[dict] | None
            The previous turn to learn from, if validation failed.
        feedback : str | None
            Validation feedback string.
        target_duration_sec : float | None
            Exact target duration (seconds) the conversation should aim for,
            drawn from the Gaussian in :func:`sample_target_duration_sec`.
        **profile
            Language, emotion, accent, gender_pair, etc.
        """
        return self.conversation_agent.run(
            title=topic["title"],
            context=topic.get("context", ""),
            conversation_type=topic.get("conversation_type"),
            previous_turns=previous_turns,
            feedback=feedback,
            target_duration_sec=target_duration_sec,
            **profile,
        )

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
        """Run the full pipeline: topic → conversation → manual validation → agent validation.

        Regenerates the conversation up to `max_agent_attempts` times based on
        agent validation. For each agent attempt, it retries up to `max_manual_attempts`
        times based on manual validation.

        Returns
        -------
        dict with keys:
            topic : dict        — output of stage 1
            turns : list[dict]  — output of stage 2 (last attempt)
            manual_validation : ValidationReport — output of manual validation
            agent_validation : AgentValidationReport | None — output of agent validation
            passed : bool       — True if final attempt passed all checks
        """
        Logger.step(f"Starting pipeline for language: {profile.get('language', 'unknown')}")
        topic = self.generate_topic(**profile)

        # Draw one exact target duration for this conversation from the Gaussian
        # so lengths follow the intended 5–10 min distribution. Sampled once and
        # reused across all retries so every regeneration aims for the same target.
        target_duration_sec = sample_target_duration_sec()
        Logger.info(
            f"Target conversation duration: {target_duration_sec:.1f}s "
            f"({target_duration_sec / 60:.2f} min)"
        )

        turns: list[dict[str, Any]] = []
        manual_report: ValidationReport | None = None
        agent_report: AgentValidationReport | None = None

        previous_turns: list[dict[str, Any]] | None = None
        feedback: str | None = None
        agent_validation_bypassed = False

        for agent_attempt in range(1, self.max_agent_attempts + 1):
            if agent_attempt > 1:
                Logger.retry(f"Agent Validation Retry: Attempt {agent_attempt}/{self.max_agent_attempts}")
            else:
                Logger.step("Stage 2: Conversation Generation & Manual Validation")
            
            # Inner loop: Manual validation retries
            for manual_attempt in range(1, self.max_manual_attempts + 1):
                if manual_attempt > 1:
                    Logger.retry(f"Manual Validation Retry: Attempt {manual_attempt}/{self.max_manual_attempts}")
                
                Logger.info("Generating conversation turns...")
                try:
                    turns = self.generate_conversation(
                        topic,
                        previous_turns=previous_turns,
                        feedback=feedback,
                        target_duration_sec=target_duration_sec,
                        **profile
                    )
                except (ValueError, LLMError) as err:
                    Logger.warning(f"Conversation generation failed: {err}")
                    manual_report = None
                    feedback = f"Conversation generation error on the previous attempt: {err}"
                    previous_turns = None
                    continue

                Logger.info("Running deterministic manual validation...")
                manual_report = self.validate_conversation(turns)
                if not manual_report.has_errors:
                    Logger.success(f"Manual validation passed on attempt {manual_attempt}!")
                    break
                else:
                    Logger.warning(f"Manual validation failed with {len(manual_report.errors)} errors.")
                
                # Setup feedback for the next manual retry
                feedback = "Manual Validation Errors:\n" + "\n".join(
                    f"- [Turn {e.turn_id}]: {e.message}" for e in manual_report.errors
                )
                
                # Pass all previous turns so the generator has full context
                previous_turns = turns if turns else None

            if manual_report is None:
                Logger.error("Conversation generation kept failing across all manual retries. Bailing out.")
                break

            if manual_report.has_errors:
                Logger.error("Failed manual validation after all retries. Bailing out.")
                break
                
            Logger.step(f"Stage 3: LLM Agent Validation (Attempt {agent_attempt}/{self.max_agent_attempts})")
            agent_report = None
            for validation_attempt in range(1, self.max_agent_validation_retries + 1):
                try:
                    agent_report = self.agent_validator.run(turns=turns, topic=topic, **profile)
                    break
                except (ValueError, LLMError) as err:
                    Logger.warning(
                        f"Agent validation call failed "
                        f"(attempt {validation_attempt}/{self.max_agent_validation_retries}): {err}"
                    )

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
                Logger.success(f"Agent validation passed! (Realism: {agent_report.realism_score}, Match: {agent_report.corpus_match_score})", bold=True)
                break
            else:
                Logger.warning(f"Agent validation failed. Verdict: {agent_report.verdict}")
            
            # Setup feedback for the next agent validation retry
            feedback = "Agent Validation Feedback:\n"
            if agent_report.feedback:
                feedback += f"{agent_report.feedback}\n"
            if agent_report.issues:
                feedback += "\n".join(f"- ({i.severity}) [Turn {i.turn_id}]: {i.description}" for i in agent_report.issues)
            
            # Pass all previous turns so the generator has full context
            previous_turns = turns if turns else None

        return {
            "topic": topic,
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


def read_corpus_instances(corpus_path: str) -> pd.DataFrame:
    """Read the corpus instances from a JSONL file and return as a DataFrame."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    df = pd.read_json(corpus_path, lines=True)
    return df


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
    env = os.getenv("ENV", "development").strip().lower()
    is_production = env == "production"

    gemini_llm = GeminiLLM()
    runner = ConversationRunner(llm=gemini_llm, max_agent_attempts=3, max_manual_attempts=3)

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
        indices = range(1)
        max_conversations = 1

    for i in indices:
        row: dict[str, Any] = {str(k): v for k, v in corpus_df.iloc[i].to_dict().items()}
        instance = CorpusInstance.from_dict(row)
        process_instance(runner, instance, storage, checkpoint, max_conversations)

    Logger.divider()
    Logger.success("All requested instances processed.", bold=True)


if __name__ == "__main__":
    main()