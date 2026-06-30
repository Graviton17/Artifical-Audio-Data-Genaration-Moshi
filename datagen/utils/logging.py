"""Tiny logging helper (mirrors Data-Processing-Moshi/dataprep/utils/logging.py)."""

from __future__ import annotations

import logging

_INITIALIZED = False


def init_logging(verbose: bool = False) -> None:
    global _INITIALIZED
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    if not _INITIALIZED:
        # Ensure a sane default if init_logging was never called (e.g. in tests).
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(f"datagen.{name}")
