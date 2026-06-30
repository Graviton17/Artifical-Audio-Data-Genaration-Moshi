"""Command-line entrypoint.

    python -m datagen.cli --input conversations --config config.yaml

Or (re)build just the manifest from an already-produced output dir:

    python -m datagen.cli --manifest-only --out dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .manifest import build_manifest
from .runner import process_dir
from .utils.logging import init_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic two-speaker conversation dataset builder for Moshi"
    )
    parser.add_argument("--input", type=Path, help="Directory of conversation .json scripts")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--out", type=Path, default=None, help="Override output directory")
    parser.add_argument("--cache", type=Path, default=None, help="Override cache directory")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only (re)build the manifest from --out")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    init_logging(args.verbose)

    config = Config.from_yaml(args.config) if args.config else Config()
    if args.out:
        config.out_dir = str(args.out)
    if args.cache:
        config.cache_dir = str(args.cache)
    if args.device:
        config.device = args.device
    if args.input:
        config.input_dir = str(args.input)

    if args.manifest_only:
        path = build_manifest(config.out_dir)
        print(f"Wrote manifest: {path}")
        return

    process_dir(config, Path(config.input_dir))


if __name__ == "__main__":
    main()
