"""Loading / reading helpers for the conversation-generation pipeline.

Small, dependency-light functions that pull configuration and corpus data in from
disk, kept separate from both the per-conversation pipeline
(:mod:`conversations_generator.runner`) and the batch driver
(:mod:`conversations_generator.corpus_runner`) so those modules stay focused on
orchestration rather than I/O.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


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


def read_corpus_instances(corpus_path: str) -> pd.DataFrame:
    """Read the corpus instances from a JSONL file and return as a DataFrame."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    df = pd.read_json(corpus_path, lines=True)
    return df
