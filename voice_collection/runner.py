"""CLI entry point for the voice-collection pipeline.

    python -m voice_collection.runner --language=hindi
    python -m voice_collection.runner --language=english --dry-run
    python -m voice_collection.runner --language=all --max-speakers=5

``--language`` selects which dataset to process (``english`` -> Svarah,
``hindi`` -> IndicVoices-R, ``all`` -> both, in order). ``--dry-run`` runs the
full fetch/filter/export stage but skips the upload step. ``--max-speakers``
overrides ``MAX_SPEAKERS_PER_LANGUAGE`` from config.json, handy for a quick
smoke test before committing to a full run.
"""
from __future__ import annotations

import argparse

from . import configuration_reader as config
from .logger import Logger
from .pipeline import VoiceCollectionPipeline

SUPPORTED_LANGUAGES = ("english", "hindi")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch unique speakers from Svarah (English) / IndicVoices-R (Hindi), "
            "keep one best-duration clip per speaker, and publish them to "
            "hf://buckets/inavlabs/voice_collection."
        )
    )
    parser.add_argument(
        "--language",
        choices=[*SUPPORTED_LANGUAGES, "all"],
        default="all",
        help="Which dataset to process. Default: all (english then hindi).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run selection + local export but skip the upload step.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Cap the number of speakers exported per language (overrides config.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    config.apply_to_environ()
    args = _parse_args(argv)

    languages = list(SUPPORTED_LANGUAGES) if args.language == "all" else [args.language]
    upload = (not args.dry_run) and config.is_upload_enabled()
    if args.dry_run:
        Logger.warning("--dry-run: selection + local export only, nothing will be uploaded.")

    pipeline = VoiceCollectionPipeline()
    if args.max_speakers is not None:
        pipeline.max_speakers = args.max_speakers

    exit_code = 0
    for language in languages:
        try:
            pipeline.run(language, upload=upload)
        except config.ConfigurationError as err:
            Logger.error(f"Configuration error for '{language}': {err}")
            exit_code = 1
        except Exception as err:  # noqa: BLE001 - keep processing remaining languages
            Logger.error(f"Pipeline failed for '{language}': {err}")
            exit_code = 1

    Logger.divider()
    if exit_code == 0:
        Logger.success("All requested languages processed.", bold=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
