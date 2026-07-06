"""Turns ``--model`` / ``--validation`` CLI arguments into concrete LLMs.

This is the single place that decides *which provider* backs each pipeline
role, so the rest of the code only ever depends on :class:`BaseLLM`.

Roles
-----
* **Generation** (``--model``): topic + conversation *content* only.
  ``--model`` always wins when given. Only when it's *omitted* does Hindi
  default to Sarvam (tuned for Indian languages); other languages default to
  the self-hosted Gemma-4 (``gemma4_local``) in that case.
* **Formatting + agent validation** (``--validation-model``): formatter agent
  and LLM validator. Defaults to the self-hosted Gemma-4 (``gemma4_local``).
  Language routing does **not** apply — validation stays on whatever
  ``--validation-model`` says.

    from conversations_generator.llm import LLMProvider, create_llm

    gen = create_llm(model="gemini", language="Hindi")      # -> GeminiLLM (explicit model wins)
    gen = create_llm(model=None, language="Hindi")          # -> SarvamLLM (default for Hindi)
    val = create_llm(model=None, apply_language_routing=False)  # -> Gemma4LocalLLM (default)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..configuration_reader import get_model, get_temperature
from .base_llm import BaseLLM
from .gemini_llm import GeminiLLM
from .gemma4_local_llm import Gemma4LocalLLM
from .groq_llm import GroqLLM
from .inception_llm import InceptionLLM
from .krutrim_llm import KrutrimLLM
from .openai_llm import OpenAILLM
from .sarvam_llm import SarvamLLM


class LLMProvider(str, Enum):
    """Every LLM backend the pipeline can construct, keyed by CLI value."""

    KRUTRIM = "krutrim"
    GEMMA = "gemma"  # Krutrim-hosted Gemma (default for --validation)
    GEMMA4_LOCAL = "gemma4_local"  # self-hosted Gemma-4 llama-server via ngrok
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
    LLMProvider.GEMMA4_LOCAL: Gemma4LocalLLM,
    LLMProvider.SARVAM: SarvamLLM,
    LLMProvider.GEMINI: GeminiLLM,
    LLMProvider.GROQ: GroqLLM,
    LLMProvider.OPENAI: OpenAILLM,
    LLMProvider.INCEPTION: InceptionLLM,
}

# Corpus languages (case-insensitive) that are always routed to Sarvam for
# *generation* only — Sarvam's models are purpose-built for Indian languages.
_SARVAM_FORCED_LANGUAGES = {"hindi"}

DEFAULT_GENERATION_PROVIDER = LLMProvider.GEMMA4_LOCAL
DEFAULT_VALIDATION_PROVIDER = LLMProvider.GEMMA4_LOCAL

# Each provider class's own hard-coded default model, used as the fallback when
# config.json's "MODELS" section has no entry for that provider.
_FALLBACK_MODELS: dict[LLMProvider, str] = {
    LLMProvider.KRUTRIM: "gemma-4-26B-A4B-it",
    LLMProvider.GEMMA: "gemma-4-26B-A4B-it",
    LLMProvider.GEMMA4_LOCAL: "gemma-4",
    LLMProvider.SARVAM: "sarvam-30b",
    LLMProvider.GEMINI: "gemini-3.5-flash",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.OPENAI: "gpt-4.1",
    LLMProvider.INCEPTION: "mercury-2",
}


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
        Provider name from ``--model`` / ``--validation-model``, or ``None``.
    language :
        Corpus language. Only consulted when ``apply_language_routing`` is True.
    apply_language_routing :
        When True (generation path) and ``model`` is ``None``, Hindi resolves
        to Sarvam by default. An explicit ``model`` always wins over this,
        for any language. When False (formatting/validation path), language
        is never consulted.
    default :
        Provider used when ``model`` is ``None``. Falls back to
        :data:`DEFAULT_GENERATION_PROVIDER` if omitted.
    """
    if (
        apply_language_routing
        and model is None
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
    chosen provider's constructor. Unless the caller explicitly overrides them,
    ``model`` and ``temperature`` are pulled from ``config.json``'s "MODELS"
    section and "TEMPERATURE" key (default 0.3), so every provider's model
    name and the shared temperature are configurable in one place.
    """
    provider = resolve_provider(
        model,
        language,
        apply_language_routing=apply_language_routing,
        default=default,
    )
    provider_cls = _PROVIDER_CLASSES[provider]
    kwargs.setdefault(
        "model", get_model(provider.value, _FALLBACK_MODELS[provider])
    )
    kwargs.setdefault("temperature", get_temperature())
    return provider_cls(**kwargs)
