"""Turns ``--model`` / ``--validation`` CLI arguments into concrete LLMs.

This is the single place that decides *which provider* backs each pipeline
role, so the rest of the code only ever depends on :class:`BaseLLM`.

Roles
-----
* **Generation** (``--model``): topic + conversation *content* only.
  Hindi instances are always forced to Sarvam; other languages use ``--model``
  if given, else Krutrim.
* **Formatting + agent validation** (``--validation``): formatter agent and
  LLM validator. Defaults to Gemma (Krutrim-hosted). Language routing does
  **not** apply — validation stays on whatever ``--validation`` says.

    from conversations_generator.llm import LLMProvider, create_llm

    gen = create_llm(model="gemini", language="English")   # -> GeminiLLM
    gen = create_llm(model="gemini", language="Hindi")      # -> SarvamLLM (forced)
    val = create_llm(model="gemma", apply_language_routing=False)  # -> KrutrimLLM (Gemma)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .base_llm import BaseLLM
from .gemini_llm import GeminiLLM
from .groq_llm import GroqLLM
from .inception_llm import InceptionLLM
from .krutrim_llm import KrutrimLLM
from .openai_llm import OpenAILLM
from .sarvam_llm import SarvamLLM


class LLMProvider(str, Enum):
    """Every LLM backend the pipeline can construct, keyed by CLI value."""

    KRUTRIM = "krutrim"
    GEMMA = "gemma"  # Krutrim-hosted Gemma (default for --validation)
    SARVAM = "sarvam"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"
    INCEPTION = "inception"

    def __str__(self) -> str:  # nicer argparse --help / error text than repr
        return self.value


# Gemma is served through Krutrim's API (default model id on KrutrimLLM).
_PROVIDER_CLASSES: dict[LLMProvider, type[BaseLLM]] = {
    LLMProvider.KRUTRIM: KrutrimLLM,
    LLMProvider.GEMMA: KrutrimLLM,
    LLMProvider.SARVAM: SarvamLLM,
    LLMProvider.GEMINI: GeminiLLM,
    LLMProvider.GROQ: GroqLLM,
    LLMProvider.OPENAI: OpenAILLM,
    LLMProvider.INCEPTION: InceptionLLM,
}

# Corpus languages (case-insensitive) that are always routed to Sarvam for
# *generation* only — Sarvam's models are purpose-built for Indian languages.
_SARVAM_FORCED_LANGUAGES = {"hindi"}

DEFAULT_GENERATION_PROVIDER = LLMProvider.KRUTRIM
DEFAULT_VALIDATION_PROVIDER = LLMProvider.GEMMA


def resolve_provider(
    model: str | LLMProvider | None,
    language: str | None = None,
    *,
    apply_language_routing: bool = True,
    default: LLMProvider | None = None,
) -> LLMProvider:
    """Pick the :class:`LLMProvider` for a CLI provider argument.

    Parameters
    ----------
    model :
        Provider name from ``--model`` / ``--validation``, or ``None``.
    language :
        Corpus language. Only consulted when ``apply_language_routing`` is True.
    apply_language_routing :
        When True (generation path), Hindi always resolves to Sarvam.
        When False (formatting/validation path), ``model``/``default`` win.
    default :
        Provider used when ``model`` is ``None``. Falls back to
        :data:`DEFAULT_GENERATION_PROVIDER` if omitted.
    """
    if (
        apply_language_routing
        and language
        and language.strip().lower() in _SARVAM_FORCED_LANGUAGES
    ):
        return LLMProvider.SARVAM

    if model is None:
        return default if default is not None else DEFAULT_GENERATION_PROVIDER
    if isinstance(model, LLMProvider):
        return model

    key = str(model).strip().lower()
    try:
        return LLMProvider(key)
    except ValueError as err:
        valid = ", ".join(p.value for p in LLMProvider)
        raise ValueError(f"Unknown LLM provider {model!r}. Choose from: {valid}") from err


def create_llm(
    model: str | LLMProvider | None = None,
    language: str | None = None,
    *,
    apply_language_routing: bool = True,
    default: LLMProvider | None = None,
    **kwargs: Any,
) -> BaseLLM:
    """Resolve and instantiate the right :class:`BaseLLM`.

    ``**kwargs`` (api_key, temperature, max_tokens, ...) are forwarded to the
    chosen provider's constructor.
    """
    provider = resolve_provider(
        model,
        language,
        apply_language_routing=apply_language_routing,
        default=default,
    )
    provider_cls = _PROVIDER_CLASSES[provider]
    return provider_cls(**kwargs)
