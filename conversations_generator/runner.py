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

import os
import pandas as pd
from pathlib import Path
from typing import Any

from .agents import ConversationGeneratorAgent, TopicGeneratorAgent
from .agents.conversation_validator_agent import ConversationValidatorAgent, AgentValidationReport
from .agents.conversation_validator_manual import ConversationValidatorManual, ValidationReport
from .llm import BaseLLM, GeminiLLM, LLMError
from .logger import Logger
from .models import CorpusInstance


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
        **profile
            Language, emotion, accent, gender_pair, etc.
        """
        return self.conversation_agent.run(
            title=topic["title"],
            context=topic.get("context", ""),
            conversation_type=topic.get("conversation_type"),
            previous_turns=previous_turns,
            feedback=feedback,
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
                
                # Pass only ONE previous turn as requested by the user
                problematic_turn = None
                if manual_report.errors and manual_report.errors[0].turn_id:
                    problematic_turn = next((t for t in turns if t.get("turn_id") == manual_report.errors[0].turn_id), turns[-1])
                if not problematic_turn and turns:
                    problematic_turn = turns[-1]
                previous_turns = [problematic_turn] if problematic_turn else None

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
            
            # Pass only ONE previous turn as requested by the user
            problematic_turn = None
            if agent_report.issues and agent_report.issues[0].turn_id:
                problematic_turn = next((t for t in turns if t.get("turn_id") == agent_report.issues[0].turn_id), turns[-1])
            if not problematic_turn and turns:
                problematic_turn = turns[-1]
            previous_turns = [problematic_turn] if problematic_turn else None

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


def main() -> None:
    load_env()  # pull GROQ_API_KEY / GEMINI_API_KEY from .env
    
    gemini_llm = GeminiLLM()
    runner = ConversationRunner(llm=gemini_llm, max_agent_attempts=3, max_manual_attempts=3)

    corpus_path = Path(__file__).resolve().parent / "data" / "corpus_instances.jsonl"
    corpus_df = read_corpus_instances(str(corpus_path))

    # Pick one row and convert to a typed CorpusInstance.
    row = corpus_df.iloc[135].to_dict()
    instance = CorpusInstance.from_dict(row)

    num_outputs = 2  # Loop to get multiple outputs for that 1 instance
    for i in range(num_outputs):
        Logger.divider()
        Logger.info(f"--- Iteration {i+1}/{num_outputs} for Instance ---")
        result = runner.run(**instance.to_profile())
        topic, turns = result["topic"], result["turns"]
        manual_report = result["manual_validation"]
        agent_report = result["agent_validation"]

        Logger.divider()
        print(
            f"[Language: {instance.language} | Gender: {instance.gender_pair} | "
            f"Type: {topic.get('conversation_type', 'unknown')}] {topic['title']}"
        )
        print(f"Context: {topic.get('context', '')}")
        print(turns)
        
        Logger.divider()
        print("--- Manual validation ---")
        if manual_report:
            manual_report.print()
            
        Logger.divider()
        print("--- Agent validation ---")
        if agent_report:
            agent_report.print()

        if not result["passed"]:
            Logger.error(f"Conversation failed validation after all retry attempts on iteration {i+1}.")
        else:
            Logger.success(f"Pipeline completed successfully for iteration {i+1}!", bold=True)
        Logger.divider()


if __name__ == "__main__":
    main()