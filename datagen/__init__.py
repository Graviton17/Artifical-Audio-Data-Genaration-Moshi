"""datagen: synthetic two-speaker conversational audio for Moshi training.

The inverse of Data-Processing-Moshi: instead of turning a real podcast into a
two-stream Moshi dataset, it turns a written two-person conversation script into
the *same* output contract (stereo wav + alignments json + dataset.jsonl), so
both feed moshi-finetune interchangeably.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
