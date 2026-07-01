"""Orchestration for the conversation-generation pipeline.

A single :class:`BaseLLM` is created once and shared by every agent, so switching
provider (Groq by default) is one argument. Today the runner drives one stage —
topic generation — but it's structured so later stages (conversation generation,
validation) slot in as new methods on :class:`ConversationRunner` without
changing how callers wire things up.

    python -m conversations_generator.runner
"""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
from typing import Any

from .agents import TopicGeneratorAgent
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
    """Drives the pipeline agents over a shared LLM, stage by stage."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        # Shared LLM: None lets each agent fall back to its own default (Groq).
        self.llm = llm
        self.topic_agent = TopicGeneratorAgent(llm)
        # Future stages (conversation generator, validators) are added here as
        # they are implemented and exposed via their own run_* method below.

    # ------------------------------------------------------------------ #
    # Stage 1: topic
    # ------------------------------------------------------------------ #
    def generate_topic(self, **profile: Any) -> dict[str, str]:
        """Produce the next single topic (see ``TopicGeneratorAgent.run``)."""
        return self.topic_agent.run(**profile)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def run(self, **profile: Any) -> dict[str, str]:
        """Run the currently-wired pipeline for one item and return its output."""
        return self.generate_topic(**profile)
    
def read_corpus_instances(corpus_path: str) -> pd.DataFrame:
    """Read the corpus instances from a JSONL file and return as a DataFrame."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    df = pd.read_json(corpus_path, lines=True)
    return df


def main() -> None:
    load_env()  # pull GROQ_API_KEY / GEMINI_API_KEY from .env
    runner = ConversationRunner()

    corpus_path = Path(__file__).resolve().parent / "data" / "corpus_instances.jsonl"
    corpus_df = read_corpus_instances(str(corpus_path))

    # Pick one row and convert to a typed CorpusInstance.
    row = corpus_df.iloc[148].to_dict()
    instance = CorpusInstance.from_dict(row)

    # .to_profile() returns only the kwargs the pipeline cares about.
    for i in range(1, 31):
        topic = runner.run(**instance.to_profile())
        print(f"[Run {i}/30 | {instance.language} | {instance.gender_pair} | {topic.get('conversation_type', 'unknown')}] "
              f"{topic['title']}\n{topic['context']}\n")


if __name__ == "__main__":
    main()
