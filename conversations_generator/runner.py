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
from .agents.conversation_validator_manual import ConversationValidatorManual, ValidationReport
from .llm import BaseLLM
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
        max_generation_attempts: int = 1,
    ) -> None:
        self.llm = llm
        self.topic_agent = TopicGeneratorAgent(llm)
        self.conversation_agent = ConversationGeneratorAgent(llm)
        self.validator = validator or ConversationValidatorManual()
        self.max_generation_attempts = max(1, max_generation_attempts)

    # ------------------------------------------------------------------ #
    # Stage 1: topic
    # ------------------------------------------------------------------ #
    def generate_topic(self, **profile: Any) -> dict[str, str]:
        """Produce the next single topic (see ``TopicGeneratorAgent.run``)."""
        return self.topic_agent.run(**profile)

    # ------------------------------------------------------------------ #
    # Stage 2: conversation
    # ------------------------------------------------------------------ #
    def generate_conversation(
        self,
        topic: dict[str, str],
        **profile: Any,
    ) -> list[dict[str, Any]]:
        """Generate a full conversation from a topic dict.

        Parameters
        ----------
        topic : dict
            Must contain ``title`` and ``context`` (output of stage 1).
        **profile
            Language, emotion, accent, gender_pair, etc.
        """
        return self.conversation_agent.run(
            title=topic["title"],
            context=topic.get("context", ""),
            conversation_type=topic.get("conversation_type"),
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
        """Run the full pipeline: topic → conversation → manual validation.

        Regenerates the conversation (same topic) up to
        ``max_generation_attempts`` times if validation reports errors, then
        returns the best attempt regardless of outcome so the caller can
        decide what to do (log, discard, escalate, etc.).

        Returns
        -------
        dict with keys:
            topic : dict        — output of stage 1
            turns : list[dict]  — output of stage 2 (last attempt)
            validation : ValidationReport — output of stage 3 (last attempt)
            attempts : int      — how many generation attempts were made
            passed : bool       — True if the final attempt had no errors
        """
        topic = self.generate_topic(**profile)

        turns: list[dict[str, Any]] = []
        report: ValidationReport | None = None

        for attempt in range(1, self.max_generation_attempts + 1):
            turns = self.generate_conversation(topic, **profile)
            report = self.validate_conversation(turns)
            if not report.has_errors:
                break

        assert report is not None  # at least one attempt always runs
        return {
            "topic": topic,
            "turns": turns,
            "validation": report,
            "attempts": attempt,
            "passed": not report.has_errors,
        }


def read_corpus_instances(corpus_path: str) -> pd.DataFrame:
    """Read the corpus instances from a JSONL file and return as a DataFrame."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    df = pd.read_json(corpus_path, lines=True)
    return df


def main() -> None:
    load_env()  # pull GROQ_API_KEY / GEMINI_API_KEY from .env
    runner = ConversationRunner(max_generation_attempts=3)

    corpus_path = Path(__file__).resolve().parent / "data" / "corpus_instances.jsonl"
    corpus_df = read_corpus_instances(str(corpus_path))

    # Pick one row and convert to a typed CorpusInstance.
    row = corpus_df.iloc[148].to_dict()
    instance = CorpusInstance.from_dict(row)

    result = runner.run(**instance.to_profile())
    topic, turns, report = result["topic"], result["turns"], result["validation"]

    print(
        f"[Language: {instance.language} | Gender: {instance.gender_pair} | "
        f"Type: {topic.get('conversation_type', 'unknown')}] {topic['title']}"
    )
    print(turns)
    print()
    print(f"--- Manual validation (attempt {result['attempts']}/{runner.max_generation_attempts}) ---")
    report.print()

    if not result["passed"]:
        print("\n⚠️  Conversation still failed manual validation after all retry attempts.")


if __name__ == "__main__":
    main()