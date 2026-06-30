"""Drive the pipeline over a directory of conversation JSON scripts.

Builds the pipeline once (loading TTS / aligner models a single time), runs every
``*.json`` script through it, writes the dataset manifest, and emits a
``report.json`` summarising what was produced or dropped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Config
from .factory import build_pipeline
from .manifest import build_manifest
from .models import GenContext
from .utils.logging import get_logger

log = get_logger("runner")


def process_dir(config: Config, input_dir: Path) -> Path:
    input_dir = Path(input_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.cache_dir)

    scripts = sorted(input_dir.glob("*.json"))
    if not scripts:
        log.warning("no .json conversation scripts found in %s", input_dir)

    pipeline = build_pipeline(config)
    log.info("pipeline: %s", pipeline)

    results = []
    t0 = time.perf_counter()
    for path in scripts:
        ctx = GenContext(
            script_path=path,
            sample_rate=config.sample_rate,
            cache_dir=cache_dir,
            out_dir=out_dir,
        )
        try:
            ctx = pipeline.run(ctx)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            log.exception("unhandled error on %s", path.name)
            results.append({"script": path.name, "dropped": True, "reason": f"error: {exc}"})
            continue

        results.append(
            {
                "script": path.name,
                "conversation_id": ctx.stem,
                "dropped": ctx.dropped,
                "reason": ctx.drop_reason,
                "duration": ctx.metadata.get("duration"),
                "variants": [v.name for v in ctx.variants if v.out_wav is not None],
            }
        )

    manifest = build_manifest(out_dir)
    report = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "num_scripts": len(scripts),
        "num_produced": sum(1 for r in results if not r["dropped"]),
        "num_dropped": sum(1 for r in results if r["dropped"]),
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log.info(
        "done: %d produced, %d dropped -> %s",
        report["num_produced"],
        report["num_dropped"],
        manifest,
    )
    return manifest
