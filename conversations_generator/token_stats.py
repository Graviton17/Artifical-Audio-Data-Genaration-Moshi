"""Aggregate LLM token usage and cost from HuggingFace bucket conversations."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .configuration_reader import (
    build_model_to_provider_map,
    get as config_get,
    get_model,
    get_provider_pricing,
)
from .logger import BOLD, DIM, RESET
from .storage.base_storage import BaseStorage, SUPPORTED_LANGUAGE_FOLDERS

try:
    from huggingface_hub import batch_bucket_files, download_bucket_files, list_bucket_tree
except ImportError:  # pragma: no cover
    batch_bucket_files = None
    download_bucket_files = None
    list_bucket_tree = None

# Terminal palette for --tokenstats (extends logger colors).
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_WHITE = "\033[97m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_MAGENTA = "\033[95m"

_MODELS_SECTION_RE = re.compile(
    r"^## Models\b.*?(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)
_USAGE_SECTION_RE = re.compile(
    r"^## LLM usage\b.*?(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)
_AGENT_LINE_RE = re.compile(
    r"^(?P<agent>[^:]+):\s*calls=(?P<calls>\d+),\s*"
    r"in=(?P<in>\d+),\s*out=(?P<out>\d+),\s*"
    r"total=(?P<total>\d+),\s*duration_sec=(?P<duration>[0-9.]+)"
)
_CALL_LINE_RE = re.compile(
    r"^- \[(?P<stage>[^\]]+)\]\s+(?P<agent>[^\s(]+)\s+\((?P<model>[^)]+)\):\s*"
    r"in=(?P<in>\d+),\s*out=(?P<out>\d+),\s*duration_sec=(?P<duration>[0-9.]+)"
)

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "hf_token_stats"

_GENERATION_AGENTS = frozenset({"topic", "conversation"})
_VALIDATION_AGENTS = frozenset(
    {"formatter", "content_validator", "format_validator", "editor"}
)


@dataclass
class ModelInfo:
    """Provider + model id used for generation and validation."""

    generation_provider: str
    generation_model: str
    validation_provider: str
    validation_model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "generation_provider": self.generation_provider,
            "generation_model": self.generation_model,
            "validation_provider": self.validation_provider,
            "validation_model": self.validation_model,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ModelInfo | None":
        if not raw:
            return None
        gen_p = raw.get("generation_provider")
        val_p = raw.get("validation_provider")
        if not gen_p or not val_p:
            return None
        return cls(
            generation_provider=str(gen_p),
            generation_model=str(raw.get("generation_model") or ""),
            validation_provider=str(val_p),
            validation_model=str(raw.get("validation_model") or ""),
        )


def legacy_krutrim_models() -> ModelInfo:
    """Model metadata for runs that used Krutrim for both generation and validation."""
    model_id = get_model("krutrim", "gemma-4-26B-A4B-it")
    return ModelInfo(
        generation_provider="krutrim",
        generation_model=model_id,
        validation_provider="krutrim",
        validation_model=model_id,
    )


def models_metadata_lines(models: ModelInfo | dict[str, str]) -> list[str]:
    data = models.to_dict() if isinstance(models, ModelInfo) else models
    return [
        "## Models",
        f"generation_provider: {data.get('generation_provider', '')}",
        f"generation_model: {data.get('generation_model', '')}",
        f"validation_provider: {data.get('validation_provider', '')}",
        f"validation_model: {data.get('validation_model', '')}",
        "",
    ]


def inject_models_into_metadata(metadata_text: str, models: ModelInfo | dict[str, str]) -> str:
    """Insert or replace the ``## Models`` block in a metadata.txt body."""
    block = "\n".join(models_metadata_lines(models))
    if _MODELS_SECTION_RE.search(metadata_text):
        return _MODELS_SECTION_RE.sub(block, metadata_text, count=1)
    if "## LLM usage" in metadata_text:
        return metadata_text.replace("## LLM usage", block + "## LLM usage", 1)
    return metadata_text.rstrip() + "\n\n" + block


@dataclass
class ConversationRecord:
    """One conversation.json loaded from the bucket."""

    path: str
    language: str
    corpus_combination_id: int | None
    index: int | None
    duration_sec: float | None
    usage: dict[str, Any] | None
    models: ModelInfo | None
    passed: bool | None

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "corpus_combination_id": self.corpus_combination_id,
            "index": self.index,
            "duration_sec": self.duration_sec,
            "usage": self.usage,
            "models": self.models.to_dict() if self.models else None,
            "passed": self.passed,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "ConversationRecord":
        return cls(
            path=str(data["path"]),
            language=str(data["language"]),
            corpus_combination_id=data.get("corpus_combination_id"),
            index=data.get("index"),
            duration_sec=data.get("duration_sec"),
            usage=data.get("usage"),
            models=ModelInfo.from_dict(data.get("models")),
            passed=data.get("passed"),
        )


@dataclass
class TokenStatsReport:
    """Aggregated token usage across many conversations."""

    conversations: int = 0
    conversations_with_usage: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_duration_sec: float = 0.0
    total_audio_duration_sec: float = 0.0
    by_language: dict[str, dict[str, int | float]] = field(default_factory=dict)
    by_agent: dict[str, dict[str, int | float]] = field(default_factory=dict)
    by_model: dict[str, dict[str, int | float]] = field(default_factory=dict)
    by_conversation: list[dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class CostStatsReport:
    """Aggregated INR cost (where pricing is configured)."""

    currency: str = "INR"
    total_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    priced_calls: int = 0
    unpriced_calls: int = 0
    by_provider: dict[str, dict[str, float | int]] = field(default_factory=dict)
    by_conversation: list[dict[str, Any]] = field(default_factory=list)


def _bucket_id() -> str:
    raw = config_get("HF_BUCKET") or ""
    return raw.removeprefix("hf://buckets/").strip("/")


def _language_prefixes(language: str | None) -> list[str]:
    if language:
        return [BaseStorage.normalize_language(language)]
    return list(SUPPORTED_LANGUAGE_FOLDERS)


def _cache_bucket_dir(bucket_id: str) -> Path:
    return _CACHE_ROOT / bucket_id.replace("/", "__")


def _cache_index_path(bucket_id: str, language: str | None) -> Path:
    lang_key = BaseStorage.normalize_language(language) if language else "all"
    return _cache_bucket_dir(bucket_id) / f"{lang_key}.index.json"


def _cache_data_path(bucket_id: str, remote_path: str) -> Path:
    safe = remote_path.replace("/", "__")
    return _cache_bucket_dir(bucket_id) / "data" / f"{safe}.json"


def _load_cache_index(bucket_id: str, language: str | None) -> dict[str, Any]:
    path = _cache_index_path(bucket_id, language)
    if not path.is_file():
        return {"files": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"files": {}}


def _save_cache_index(bucket_id: str, language: str | None, index: dict[str, Any]) -> None:
    path = _cache_index_path(bucket_id, language)
    path.parent.mkdir(parents=True, exist_ok=True)
    index["bucket_id"] = bucket_id
    index["language"] = language or "all"
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def iter_hf_metadata_files(
    language: str | None = None,
    *,
    bucket_id: str | None = None,
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(language, path, xet_hash)`` for every ``metadata.txt``."""
    if list_bucket_tree is None:
        raise ImportError("huggingface_hub>=1.5.0 is required.")

    bucket = bucket_id or _bucket_id()
    prefixes = set(_language_prefixes(language))

    for item in list_bucket_tree(bucket, recursive=True):
        path = getattr(item, "path", None)
        xet_hash = getattr(item, "xet_hash", None)
        if not path or not xet_hash or not path.endswith("metadata.txt"):
            continue
        parts = path.split("/")
        if len(parts) < 4:
            continue
        lang = parts[0].lower()
        if lang not in prefixes:
            continue
        yield lang, path, str(xet_hash)


def iter_hf_conversation_files(
    language: str | None = None,
    *,
    bucket_id: str | None = None,
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(language, path, xet_hash)`` for every ``conversation.json``."""
    if list_bucket_tree is None:
        raise ImportError("huggingface_hub>=1.5.0 is required.")

    bucket = bucket_id or _bucket_id()
    prefixes = set(_language_prefixes(language))

    for item in list_bucket_tree(bucket, recursive=True):
        path = getattr(item, "path", None)
        xet_hash = getattr(item, "xet_hash", None)
        if not path or not xet_hash or not path.endswith("conversation.json"):
            continue
        parts = path.split("/")
        if len(parts) < 4:
            continue
        lang = parts[0].lower()
        if lang not in prefixes:
            continue
        yield lang, path, str(xet_hash)


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() == "none" or text == "":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_header_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("## "):
            break
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key.strip()] = _parse_scalar(value)
    return fields


def _parse_models_section(text: str) -> ModelInfo | None:
    match = _MODELS_SECTION_RE.search(text)
    if not match:
        return None
    raw: dict[str, str] = {}
    for line in match.group(0).splitlines():
        if ": " not in line or line.startswith("##"):
            continue
        key, value = line.split(": ", 1)
        raw[key.strip()] = value.strip()
    return ModelInfo.from_dict(raw)


def parse_metadata_usage(text: str) -> dict[str, Any] | None:
    """Rebuild the ``usage`` dict from a ``metadata.txt`` body."""
    match = _USAGE_SECTION_RE.search(text)
    if not match:
        return None

    section = match.group(0)
    totals: dict[str, Any] = {}
    by_agent: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []

    in_by_agent = False
    in_calls = False
    for line in section.splitlines():
        stripped = line.strip()
        if stripped == "### By agent":
            in_by_agent = True
            in_calls = False
            continue
        if stripped == "### Per call (chronological)":
            in_by_agent = False
            in_calls = True
            continue
        if stripped.startswith("##") or not stripped:
            continue

        if not in_by_agent and not in_calls and ": " in stripped:
            key, value = stripped.split(": ", 1)
            totals[key.strip()] = _parse_scalar(value)
            continue

        if in_by_agent:
            agent_match = _AGENT_LINE_RE.match(stripped)
            if agent_match:
                agent = agent_match.group("agent").strip()
                by_agent[agent] = {
                    "calls": int(agent_match.group("calls")),
                    "input_tokens": int(agent_match.group("in")),
                    "output_tokens": int(agent_match.group("out")),
                    "total_tokens": int(agent_match.group("total")),
                    "duration_sec": float(agent_match.group("duration")),
                }
            continue

        if in_calls:
            call_match = _CALL_LINE_RE.match(stripped)
            if call_match:
                stage_raw = call_match.group("stage")
                attempt = None
                if ", attempt=" in stage_raw:
                    stage, attempt_s = stage_raw.split(", attempt=", 1)
                    attempt = int(attempt_s)
                else:
                    stage = stage_raw
                in_tok = int(call_match.group("in"))
                out_tok = int(call_match.group("out"))
                calls.append(
                    {
                        "stage": stage.strip(),
                        "attempt": attempt,
                        "agent": call_match.group("agent").strip(),
                        "model": call_match.group("model").strip(),
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok,
                        "duration_sec": float(call_match.group("duration")),
                    }
                )

    if not totals and not by_agent and not calls:
        return None

    if "total_tokens" not in totals:
        totals["total_tokens"] = int(totals.get("total_input_tokens", 0) or 0) + int(
            totals.get("total_output_tokens", 0) or 0
        )

    return {"totals": totals, "by_agent": by_agent, "calls": calls}


def _parse_metadata_record(lang: str, path: str, text: str) -> ConversationRecord:
    header = _parse_header_fields(text)
    usage = parse_metadata_usage(text)
    duration = header.get("duration_sec")
    if duration is None:
        for line in text.splitlines():
            if line.startswith("duration_sec:"):
                duration = _parse_scalar(line.split(": ", 1)[1])
                break

    return ConversationRecord(
        path=path.replace("/metadata.txt", "/conversation.json"),
        language=lang,
        corpus_combination_id=header.get("corpus_combination_id"),
        index=header.get("conversation_index"),
        duration_sec=float(duration) if duration is not None else None,
        usage=usage,
        models=_parse_models_section(text),
        passed=header.get("passed"),
    )


def _parse_conversation_json(
    lang: str,
    path: str,
    data: dict[str, Any],
) -> ConversationRecord:
    return ConversationRecord(
        path=path,
        language=lang,
        corpus_combination_id=data.get("corpus_combination_id"),
        index=data.get("index"),
        duration_sec=data.get("duration_sec"),
        usage=data.get("usage"),
        models=ModelInfo.from_dict(data.get("models")),
        passed=data.get("passed"),
    )


def load_hf_conversations(
    language: str | None = None,
    *,
    bucket_id: str | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[list[ConversationRecord], int, int]:
    """Load token usage from HF ``metadata.txt`` files (not full conversation JSON).

    ``metadata.txt`` already carries the ``## LLM usage`` block written at save
    time, so token stats do not need to download heavy ``conversation.json``
    payloads. Falls back to ``conversation.json`` only when metadata lacks usage
    (older bucket objects).

    Returns ``(records, cache_hits, cache_misses)``.
    """
    if download_bucket_files is None:
        raise ImportError("huggingface_hub>=1.5.0 is required.")

    bucket = bucket_id or _bucket_id()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    remote_files = {
        path: (lang, xet_hash)
        for lang, path, xet_hash in iter_hf_metadata_files(language, bucket_id=bucket)
    }

    cache_index = _load_cache_index(bucket, language)
    cached_files: dict[str, str] = cache_index.get("files") or {}

    records: list[ConversationRecord] = []
    cache_hits = 0
    cache_misses = 0
    to_download: list[tuple[str, str, str]] = []

    for path, (lang, xet_hash) in remote_files.items():
        cache_path = _cache_data_path(bucket, path)
        if (
            use_cache
            and not refresh
            and cached_files.get(path) == xet_hash
            and cache_path.is_file()
        ):
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                records.append(ConversationRecord.from_cache_dict(cached))
                cache_hits += 1
                continue
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        to_download.append((lang, path, xet_hash))

    from tqdm import tqdm

    download_bar = tqdm(
        to_download,
        desc="Reading metadata.txt",
        unit="file",
        disable=not to_download,
        dynamic_ncols=True,
    )
    for lang, path, xet_hash in download_bar:
        local = _cache_data_path(bucket, path)
        local.parent.mkdir(parents=True, exist_ok=True)
        download_bucket_files(bucket, files=[(path, str(local))])
        text = local.read_text(encoding="utf-8")
        record = _parse_metadata_record(lang, path, text)

        if record.usage is None:
            json_path = path.replace("/metadata.txt", "/conversation.json")
            json_local = _cache_data_path(bucket, json_path)
            json_local.parent.mkdir(parents=True, exist_ok=True)
            download_bucket_files(bucket, files=[(json_path, str(json_local))])
            data = json.loads(json_local.read_text(encoding="utf-8"))
            record = _parse_conversation_json(lang, json_path, data)

        local.write_text(
            json.dumps(record.to_cache_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cached_files[path] = xet_hash
        records.append(record)
        cache_misses += 1

    # Drop cache entries removed from the bucket.
    for stale_path in set(cached_files) - set(remote_files):
        cached_files.pop(stale_path, None)
        stale_file = _cache_data_path(bucket, stale_path)
        stale_file.unlink(missing_ok=True)

    if use_cache:
        _save_cache_index(bucket, language, {"files": cached_files})

    records.sort(key=lambda r: r.path)
    return records, cache_hits, cache_misses


def _resolve_provider_for_call(
    call: dict[str, Any],
    models: ModelInfo | None,
    model_to_provider: dict[str, str],
) -> str | None:
    agent = str(call.get("agent") or "")
    if models:
        if agent in _GENERATION_AGENTS:
            return models.generation_provider.lower()
        if agent in _VALIDATION_AGENTS:
            return models.validation_provider.lower()
    model_id = str(call.get("model") or "")
    return model_to_provider.get(model_id)


def _call_cost_inr(
    provider: str | None,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float, float] | None:
    if not provider:
        return None
    pricing = get_provider_pricing(provider)
    if not pricing:
        return None
    in_rate = pricing.get("input_per_1m")
    out_rate = pricing.get("output_per_1m")
    if in_rate is None and out_rate is None:
        return None
    in_cost = (input_tokens / 1_000_000) * float(in_rate or 0)
    out_cost = (output_tokens / 1_000_000) * float(out_rate or 0)
    return in_cost, out_cost, in_cost + out_cost


def compute_cost_stats(records: list[ConversationRecord]) -> CostStatsReport:
    """Compute INR costs from per-call usage + ``MODEL_PRICING`` in config."""
    report = CostStatsReport()
    model_to_provider = build_model_to_provider_map()

    for rec in records:
        usage = rec.usage or {}
        conv_in_cost = 0.0
        conv_out_cost = 0.0
        conv_total = 0.0
        conv_priced = 0
        conv_unpriced = 0

        for call in usage.get("calls") or []:
            in_tok = int(call.get("input_tokens", 0) or 0)
            out_tok = int(call.get("output_tokens", 0) or 0)
            provider = _resolve_provider_for_call(call, rec.models, model_to_provider)
            costs = _call_cost_inr(provider, in_tok, out_tok)
            if costs is None:
                conv_unpriced += 1
                report.unpriced_calls += 1
                continue
            in_cost, out_cost, total = costs
            conv_in_cost += in_cost
            conv_out_cost += out_cost
            conv_total += total
            conv_priced += 1
            report.priced_calls += 1
            report.input_cost += in_cost
            report.output_cost += out_cost
            report.total_cost += total

            if provider:
                bucket = report.by_provider.setdefault(
                    provider,
                    {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "input_cost": 0.0,
                        "output_cost": 0.0,
                        "total_cost": 0.0,
                    },
                )
                bucket["calls"] = int(bucket["calls"]) + 1
                bucket["input_tokens"] = int(bucket["input_tokens"]) + in_tok
                bucket["output_tokens"] = int(bucket["output_tokens"]) + out_tok
                bucket["input_cost"] = float(bucket["input_cost"]) + in_cost
                bucket["output_cost"] = float(bucket["output_cost"]) + out_cost
                bucket["total_cost"] = float(bucket["total_cost"]) + total

        if conv_priced:
            report.by_conversation.append(
                {
                    "path": rec.path,
                    "input_cost": conv_in_cost,
                    "output_cost": conv_out_cost,
                    "total_cost": conv_total,
                    "priced_calls": conv_priced,
                    "unpriced_calls": conv_unpriced,
                }
            )

    return report


def aggregate_token_stats(
    records: list[ConversationRecord],
    *,
    cache_hits: int = 0,
    cache_misses: int = 0,
) -> TokenStatsReport:
    """Build a summary report from loaded conversation records."""
    report = TokenStatsReport(cache_hits=cache_hits, cache_misses=cache_misses)

    for rec in records:
        report.conversations += 1
        if rec.duration_sec:
            report.total_audio_duration_sec += float(rec.duration_sec)

        lang_bucket = report.by_language.setdefault(
            rec.language,
            {
                "conversations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "llm_duration_sec": 0.0,
                "audio_duration_sec": 0.0,
            },
        )
        lang_bucket["conversations"] = int(lang_bucket["conversations"]) + 1
        if rec.duration_sec:
            lang_bucket["audio_duration_sec"] = float(lang_bucket["audio_duration_sec"]) + float(
                rec.duration_sec
            )

        usage = rec.usage or {}
        totals = usage.get("totals") or {}
        if not totals:
            continue

        report.conversations_with_usage += 1
        in_tok = int(totals.get("input_tokens", 0) or 0)
        out_tok = int(totals.get("output_tokens", 0) or 0)
        llm_sec = float(totals.get("duration_sec", 0) or 0)

        report.total_input_tokens += in_tok
        report.total_output_tokens += out_tok
        report.total_llm_duration_sec += llm_sec

        lang_bucket["input_tokens"] = int(lang_bucket["input_tokens"]) + in_tok
        lang_bucket["output_tokens"] = int(lang_bucket["output_tokens"]) + out_tok
        lang_bucket["total_tokens"] = int(lang_bucket["total_tokens"]) + in_tok + out_tok
        lang_bucket["llm_duration_sec"] = float(lang_bucket["llm_duration_sec"]) + llm_sec

        for agent, stats in (usage.get("by_agent") or {}).items():
            bucket = report.by_agent.setdefault(
                agent,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_sec": 0.0,
                },
            )
            bucket["calls"] = int(bucket["calls"]) + int(stats.get("calls", 0) or 0)
            bucket["input_tokens"] = int(bucket["input_tokens"]) + int(
                stats.get("input_tokens", 0) or 0
            )
            bucket["output_tokens"] = int(bucket["output_tokens"]) + int(
                stats.get("output_tokens", 0) or 0
            )
            bucket["total_tokens"] = int(bucket["total_tokens"]) + int(
                stats.get("total_tokens", 0) or 0
            )
            bucket["duration_sec"] = float(bucket["duration_sec"]) + float(
                stats.get("duration_sec", 0) or 0
            )

        for call in usage.get("calls") or []:
            model = str(call.get("model") or "unknown")
            prov_bucket = report.by_model.setdefault(
                model,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_sec": 0.0,
                },
            )
            prov_bucket["calls"] = int(prov_bucket["calls"]) + 1
            prov_bucket["input_tokens"] = int(prov_bucket["input_tokens"]) + int(
                call.get("input_tokens", 0) or 0
            )
            prov_bucket["output_tokens"] = int(prov_bucket["output_tokens"]) + int(
                call.get("output_tokens", 0) or 0
            )
            prov_bucket["total_tokens"] = int(prov_bucket["total_tokens"]) + int(
                call.get("total_tokens", 0) or 0
            )
            prov_bucket["duration_sec"] = float(prov_bucket["duration_sec"]) + float(
                call.get("duration_sec", 0) or 0
            )

        models = rec.models
        report.by_conversation.append(
            {
                "path": rec.path,
                "language": rec.language,
                "corpus_combination_id": rec.corpus_combination_id,
                "index": rec.index,
                "passed": rec.passed,
                "audio_duration_sec": rec.duration_sec,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "llm_duration_sec": llm_sec,
                "generation_provider": models.generation_provider if models else None,
                "validation_provider": models.validation_provider if models else None,
            }
        )

    return report


def _inr(value: float) -> str:
    return f"₹{value:,.2f}"


def _section(title: str) -> None:
    print(f"\n{_CYAN}{BOLD}  {title}{RESET}")
    print(f"{DIM}  {'─' * 66}{RESET}")


def _kv(label: str, value: str, *, color: str = _WHITE) -> None:
    print(f"  {DIM}{label:<28}{RESET}{color}{value}{RESET}")


def print_colored_token_stats(
    report: TokenStatsReport,
    cost: CostStatsReport,
    *,
    language: str | None = None,
) -> None:
    """Print a colorized, structured token + cost summary."""
    scope = language or "all languages"
    width = 70

    print()
    print(f"{_CYAN}{BOLD}{'═' * width}{RESET}")
    print(f"{_CYAN}{BOLD}  HF TOKEN & COST STATS  {DIM}({scope}){RESET}")
    print(f"{_CYAN}{BOLD}{'═' * width}{RESET}")

    if report.cache_hits or report.cache_misses:
        _section("Cache")
        _kv("Cache hits", str(report.cache_hits), color=_GREEN)
        _kv("Downloaded metadata.txt", str(report.cache_misses), color=_YELLOW if report.cache_misses else _GREEN)
        if report.cache_hits and report.cache_misses == 0:
            print(f"  {_GREEN}✓ Using local cache — no HF re-download needed{RESET}")
        elif report.cache_misses:
            print(f"  {_DIM}Token stats read lightweight metadata.txt (not conversation.json){RESET}")

    _section("Totals")
    _kv("Conversations", f"{report.conversations}", color=_WHITE)
    _kv("With usage data", f"{report.conversations_with_usage}", color=_WHITE)
    _kv("Input tokens", f"{report.total_input_tokens:,}", color=_BLUE)
    _kv("Output tokens", f"{report.total_output_tokens:,}", color=_BLUE)
    _kv("Total tokens", f"{report.total_tokens:,}", color=_BRIGHT_GREEN + BOLD)
    _kv("LLM time", f"{report.total_llm_duration_sec:,.1f}s", color=_WHITE)
    _kv("Audio generated", f"{report.total_audio_duration_sec / 60:,.2f} min", color=_WHITE)

    _section(f"Estimated cost ({cost.currency})")
    _kv("Input cost", _inr(cost.input_cost), color=_YELLOW)
    _kv("Output cost", _inr(cost.output_cost), color=_YELLOW)
    _kv("Total cost", _inr(cost.total_cost), color=_BRIGHT_YELLOW + BOLD)
    _kv("Priced LLM calls", str(cost.priced_calls), color=_GREEN)
    if cost.unpriced_calls:
        _kv("Unpriced calls", str(cost.unpriced_calls), color=_YELLOW)

    if cost.by_provider:
        _section(f"Cost by provider ({cost.currency})")
        for provider, stats in sorted(
            cost.by_provider.items(),
            key=lambda x: float(x[1]["total_cost"]),
            reverse=True,
        ):
            print(
                f"  {_MAGENTA}{BOLD}{provider:12}{RESET} "
                f"{DIM}in={_inr(float(stats['input_cost']))}{RESET}  "
                f"{DIM}out={_inr(float(stats['output_cost']))}{RESET}  "
                f"{_BRIGHT_MAGENTA}total={_inr(float(stats['total_cost']))}{RESET}  "
                f"{DIM}({int(stats['calls'])} calls){RESET}"
            )

    if report.by_agent:
        _section("Tokens by agent")
        for agent, stats in sorted(report.by_agent.items()):
            print(
                f"  {_BLUE}{agent:18}{RESET} "
                f"{DIM}calls={int(stats['calls']):>3}{RESET}  "
                f"in={_GREEN}{int(stats['input_tokens']):>8,}{RESET}  "
                f"out={_YELLOW}{int(stats['output_tokens']):>8,}{RESET}  "
                f"total={_BRIGHT_GREEN}{int(stats['total_tokens']):>9,}{RESET}"
            )

    if report.by_model:
        _section("Tokens by model id")
        for model, stats in sorted(report.by_model.items()):
            print(
                f"  {_WHITE}{model:24}{RESET} "
                f"in={int(stats['input_tokens']):>8,}  "
                f"out={int(stats['output_tokens']):>8,}  "
                f"total={int(stats['total_tokens']):>9,}"
            )

    if report.by_conversation:
        _section("Per conversation")
        cost_by_path = {row["path"]: row for row in cost.by_conversation}
        for row in sorted(
            report.by_conversation,
            key=lambda r: (r.get("language") or "", r.get("path") or ""),
        ):
            path = row["path"]
            short = "/".join(path.split("/")[-3:])
            cost_row = cost_by_path.get(path)
            cost_s = _inr(float(cost_row["total_cost"])) if cost_row else f"{DIM}n/a{RESET}"
            gen = row.get("generation_provider") or "?"
            val = row.get("validation_provider") or "?"
            print(
                f"  {_CYAN}{short}{RESET}\n"
                f"    {DIM}tokens{RESET}  "
                f"in={_GREEN}{row['input_tokens']:,}{RESET}  "
                f"out={_YELLOW}{row['output_tokens']:,}{RESET}  "
                f"total={_BRIGHT_GREEN}{row['total_tokens']:,}{RESET}  "
                f"{DIM}audio={row.get('audio_duration_sec')}s{RESET}\n"
                f"    {DIM}models{RESET}  gen={_MAGENTA}{gen}{RESET}  "
                f"val={_MAGENTA}{val}{RESET}  "
                f"{DIM}cost{RESET}  {cost_s}"
            )

    print()
    print(f"{_CYAN}{BOLD}{'═' * width}{RESET}")
    print()


def print_token_stats(
    language: str | None = None,
    *,
    refresh: bool = False,
) -> tuple[TokenStatsReport, CostStatsReport]:
    """List bucket metadata files, aggregate usage, and print summary."""
    records, cache_hits, cache_misses = load_hf_conversations(
        language,
        use_cache=True,
        refresh=refresh,
    )
    report = aggregate_token_stats(records, cache_hits=cache_hits, cache_misses=cache_misses)
    cost = compute_cost_stats(records)
    print_colored_token_stats(report, cost, language=language)
    return report, cost


def patch_hf_model_metadata(
    language: str,
    models: ModelInfo,
    *,
    bucket_id: str | None = None,
    dry_run: bool = False,
) -> int:
    """Add/update ``models`` on every conversation under ``<language>/`` on HF."""
    if batch_bucket_files is None or download_bucket_files is None:
        raise ImportError("huggingface_hub>=1.5.0 is required.")

    bucket = bucket_id or _bucket_id()
    lang = BaseStorage.normalize_language(language)
    tmp = _CACHE_ROOT / "patch_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    patched = 0

    for _, json_path, _ in iter_hf_conversation_files(lang, bucket_id=bucket):
        local_json = tmp / json_path.replace("/", "__")
        local_json.parent.mkdir(parents=True, exist_ok=True)
        download_bucket_files(bucket, files=[(json_path, str(local_json))])
        data = json.loads(local_json.read_text(encoding="utf-8"))
        data["models"] = models.to_dict()
        local_json.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        uploads: list[tuple[str, str]] = [(str(local_json), json_path)]
        meta_path = json_path.replace("/conversation.json", "/metadata.txt")
        local_meta = tmp / meta_path.replace("/", "__")
        try:
            download_bucket_files(bucket, files=[(meta_path, str(local_meta))])
            if local_meta.is_file():
                text = local_meta.read_text(encoding="utf-8")
                local_meta.write_text(
                    inject_models_into_metadata(text, models), encoding="utf-8"
                )
                uploads.append((str(local_meta), meta_path))
        except Exception:
            pass

        if dry_run:
            print(f"would patch {json_path}")
        else:
            batch_bucket_files(bucket, add=uploads)
        patched += 1

    return patched
