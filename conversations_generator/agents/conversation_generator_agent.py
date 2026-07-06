"""Agent that generates a full multi-turn conversation as tagged plain text.

This is the first half of the two-stage generation pipeline. It takes the
title + context produced by :class:`TopicGeneratorAgent` and writes a natural
two-speaker conversation as **plain text**, one turn per line, using lightweight
inline tags for turn-taking:

    S1: [normal] Toh tumne report finish kar li? (Neutral)
    S2: [overlap] -> S1@0.15 Haan main— (Neutral)
    S1: [interrupt] -> S2 —kyunki Sarah pooch rahi thi. (Angry)
    S2: [backchannel] -> S1@0.6 mhm (Neutral)
    S2: [normal] Maine kal raat hi complete kar li thi. (Happy)

Two details matter downstream, both explained in full in
``data/prompts/conversation-generator-agent.md``:

* ``[overlap]``/``[backchannel]`` references carry a ``@<ratio>`` (0.0-1.0)
  estimating how far into the partner turn's utterance this line begins, so
  the formatter can place it at that exact point instead of guessing.
* An interrupted (``—``-ending) line must contain **only the words actually
  spoken before the cut-off** — a genuine incomplete fragment, not a full
  sentence with a dash tacked on — and the interrupting line must react only
  to that fragment, never to content the victim never got to say.

It deliberately produces **no** JSON, timestamps, turn IDs, or cross-references —
those are added by :class:`~conversations_generator.agents.conversation_formater_agent.ConversationFormatterAgent`.
Keeping this agent's job small (good dialogue + correct tags) is what makes small
models reliable here.

The system prompt is managed in Langfuse under the name
``conversation-generator-agent``.
"""

from __future__ import annotations

from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

# ------------------------------------------------------------------ #
# Per-accent marker cheat-sheet, injected INTO THE USER PROMPT (not the
# system prompt) whenever a speaker's accent is not "Normal".
#
# Accent is the single most-missed corpus field: the agent validator wants a
# non-Normal accent to be *textually audible* (regional vocab, particles,
# phrasing — not just one address term), and burying that in the long system
# prompt wasn't enough. Placing the exact markers next to the request, keyed to
# the specific accent asked for, gives the small model a concrete, high-salience
# target. Markers are real words in normal Devanagari spelling — never phonetic
# fakes. Keys are matched case-insensitively against the accent string.
# ------------------------------------------------------------------ #
_ACCENT_MARKERS: dict[str, str] = {
    "bengali": (
        "address दादा (to a man) / दीदी (to a woman); frequent tag question '…ना?' "
        "and the affectionate particle '…गो' (हाँ गो, अच्छा गो); interjections 'ऐं?', "
        "'बाप रे!', 'ईश!', 'ओ बाबा'; discourse openers 'अच्छा तो…', 'सुनिए ना…', "
        "'एक बात बोलूँ…'; soft, deferential, slightly formal requests "
        "(ज़रा देख लीजिए ना, खूब बढ़िया); double 'अच्छा अच्छा'"
    ),
    "punjabi": (
        "address यार / पाजी; interjections 'ओए', 'चक दे'; sentence-final '…जी', '…ना'; "
        "openers 'अरे यार', 'सुन ना'; direct, energetic, high-emphasis phrasing "
        "(बिल्कुल यार, चंगा, ठीक है जी, कमाल कर दिया)"
    ),
    "gujarati": (
        "address भाई / बेन; interjection 'अरे वाह'; very frequent tag '…ना?' "
        "(सही है ना?, बराबर ने?); opener 'अच्छा सुनो'; practical/frugal framing "
        "(पैसा वसूल, सस्ता और टिकाऊ, भाव ठीक होना चाहिए)"
    ),
    "west": (
        "Mumbai-Hindi: address भाऊ; interjections 'अरे यार', 'क्या बे'; tag '…ना'; "
        "'ऐसा है ना'; high-energy words एकदम, झकास, कडक, बिंदास; clipped, fast sentences"
    ),
    "south indian": (
        "careful, formal Hindi (often a second language): address सर / मैडम used more "
        "than casual Hindi would; 'ओके-ओके'; tag '…है ना'; opener 'मतलब यह है कि…'; "
        "fewer contractions, occasional literal/precise phrasing"
    ),
}


# For an ENGLISH conversation a regional accent can't be shown with Hindi words
# (that breaks the English requirement) — it's carried by Indian-English syntax
# and rhythm instead. Written English doesn't reliably distinguish region, so all
# non-Normal accents share one Indian-English note here; the content validator
# grades English accent leniently to match.
_INDIAN_ENGLISH_ACCENT_NOTE = (
    "an Indian speaker of English — carry the accent through Indian-English "
    "rhythm and phrasing that is STILL English: tag questions 'na'/'no?', "
    "emphasis with 'only'/'itself', 'do one thing', 'what all', and the odd "
    "naturalised interjection ('arre', 'yaar', 'haan') used sparingly. Do NOT "
    "insert Hindi words/clauses or Devanagari — that fails the English "
    "requirement. A light, natural flavour is enough."
)


def _accent_guidance(
    agent_accent: str | None,
    user_accent: str | None,
    language: str | None,
) -> list[str]:
    """Build a user-prompt block of concrete accent guidance.

    For Hindi/Hinglish this injects the per-accent Devanagari marker cheat-sheet.
    For English it injects an Indian-English note instead (Hindi markers would
    violate the English requirement). Returns an empty list when neither speaker
    has a recognized non-Normal accent, so Normal-accent conversations are
    unaffected.
    """
    is_english = bool(language) and language.strip().lower() == "english"
    entries: list[str] = []
    for who, accent in (("Speaker 1 (agent)", agent_accent), ("Speaker 2 (user)", user_accent)):
        if not accent:
            continue
        if is_english:
            if accent.strip().lower() == "normal":
                continue
            entries.append(f"- **{who} — {accent}:** {_INDIAN_ENGLISH_ACCENT_NOTE}")
        else:
            markers = _ACCENT_MARKERS.get(accent.strip().lower())
            if markers:
                entries.append(f"- **{who} — {accent}:** {markers}.")

    if not entries:
        return []

    if is_english:
        return [
            "",
            "## Accent in English (graded — keep it English)",
            "The conversation is in English, so a regional accent shows through "
            "Indian-English phrasing, NOT Hindi words. Give each speaker below a "
            "little natural Indian-English flavour from their first turns on — a "
            "couple of markers is plenty. Do NOT code-switch into Hindi.",
            *entries,
        ]

    return [
        "",
        "## Accent markers to actually use (MANDATORY — this is graded)",
        "Plain Hindi/Hinglish with no regional flavour FAILS the accent check — one "
        "address term alone is NOT enough. Starting from this speaker's FIRST turn, "
        "weave in several of the markers below from DIFFERENT categories and reuse "
        "them naturally throughout (they also make great backchannels, which adds "
        "variety). Keep normal Devanagari/Romanized spelling — real words, not "
        "phonetic tricks. The address term must also match the addressee's gender.",
        *entries,
    ]


_LANGUAGE_DIRECTIVES: dict[str, str] = {
    "hindi": (
        "Write PREDOMINANTLY in Hindi (Devanagari script). English words are allowed "
        "ONLY for terms with no natural Hindi equivalent (e.g. brand names, \"laptop\", "
        "\"WiFi\"). Do NOT write full English sentences or clauses — every sentence's "
        "grammar and structure must be Hindi."
    ),
    "hinglish": (
        "Write genuine Hindi-English code-mixing: Hindi grammar, postpositions, and "
        "verb-final sentence structure, with English nouns/verbs swapped in at natural "
        "points. This is NOT an English sentence with a few Hindi words sprinkled in, "
        "and NOT the same clause restated in both languages back-to-back."
    ),
    "english": (
        "Write entirely in plain, natural English. Do NOT mix in Hindi words, "
        "transliterations, or Devanagari script."
    ),
}


def _language_directive(language: str | None) -> list[str]:
    """Mandatory, unambiguous instruction for the exact requested language.

    Keyed on the corpus's three supported registers (Hindi/Hinglish/English);
    unrecognized values are left to the base system prompt.
    """
    if not language:
        return []
    directive = _LANGUAGE_DIRECTIVES.get(language.strip().lower())
    if not directive:
        return []
    return [
        "",
        f"## MANDATORY language requirement: {language}",
        directive,
        "This applies to EVERY line, from the first turn to the last — do not drift "
        "into a different register partway through.",
    ]


def _number_directive(include_numbers: bool) -> list[str]:
    """Prompt block controlling whether the conversation is number-rich.

    ``include_numbers=True`` requires several concrete figures *with the
    reasoning around them*; ``False`` keeps the dialogue qualitative. Toggled
    per-conversation by the runner from ``NUMBER_INCLUSION_PERCENTAGE``.
    """
    if include_numbers:
        return [
            "",
            "## Numbers (MANDATORY for this conversation)",
            "Weave several CONCRETE numbers naturally into the dialogue — e.g. "
            "prices/amounts, dates, durations, quantities, percentages/discounts, "
            "measurements, ages, scores, or distances — AND natural reasoning "
            "around them: speakers state specific figures and then explain, "
            "justify, or compare them the way people do OUT LOUD (why a price is "
            "high, whether an option is worth it, weighing two choices), not just "
            "drop a number in isolation. **Do NOT narrate arithmetic like a "
            "calculator** — never read out a bare equation such as '45 + 200 = "
            "245'; a real speaker just says the result ('toh total do sau "
            "pentaalis ho gaya') and moves on. Spell numbers the way people "
            "actually SAY them aloud in this language (this is spoken audio data), "
            "not as bare digits where that would sound unnatural. Keep it "
            "realistic — a few well-motivated numbers across the conversation, "
            "not a figure crammed into every line.",
        ]
    return [
        "",
        "## Numbers",
        "Keep this conversation qualitative: do NOT force in statistics or "
        "specific figures. Only use a number if it is genuinely unavoidable and "
        "natural for a passing remark.",
    ]


class ConversationGeneratorAgent(BaseAgent):
    """Generate a multi-turn conversation as tagged plain text.

    Returns the raw transcript string (one turn per line). The format is defined
    in ``data/prompts/conversation-generator-agent.md``.
    """

    prompt_name = "conversation-generator-agent"
    temperature_key = "conversation"
    agent_name = "conversation"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        title: str,
        context: str,
        language: str = "Hinglish",
        conversation_type: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
        previous_transcript: str | None = None,
        feedback: str | None = None,
        target_duration_sec: float | None = None,
        include_numbers: bool = False,
        **overrides: Any,
    ) -> str:
        """Generate a full conversation for the given topic, as plain text.

        Parameters
        ----------
        title, context : str
            Topic from :class:`TopicGeneratorAgent`.
        language : str
            Language the dialogue should be written in.
        conversation_type : str | None
            Type of conversation (e.g. "Sales", "Inquiry").
        agent_emotion, user_emotion : str | None
            Dominant emotion for each speaker.
        agent_accent, user_accent : str | None
            Accent style for each speaker (TTS only, not spelled out).
        gender_pair : str | None
            Gender pair string like "M-F".
        previous_transcript : str | None
            The previous attempt's transcript to fix (used on retries).
        feedback : str | None
            Validation feedback describing what was wrong with the previous attempt.
        target_duration_sec : float | None
            Approximate target duration to pace turn count against.
        include_numbers : bool
            When True, the dialogue must weave in concrete numbers with reasoning;
            when False it stays qualitative. Toggled per-conversation by the runner
            from ``NUMBER_INCLUSION_PERCENTAGE`` in config.json.
        **overrides
            Extra kwargs forwarded to the LLM (temperature, max_tokens, etc.).

        Returns
        -------
        str
            The tagged plain-text transcript, one turn per line.
        """
        prompt = self._build_prompt(
            title=title,
            context=context,
            language=language,
            conversation_type=conversation_type,
            agent_emotion=agent_emotion,
            user_emotion=user_emotion,
            agent_accent=agent_accent,
            user_accent=user_accent,
            gender_pair=gender_pair,
            previous_transcript=previous_transcript,
            feedback=feedback,
            target_duration_sec=target_duration_sec,
            include_numbers=include_numbers,
        )

        system_vars: dict[str, Any] = {}
        if conversation_type:
            system_vars["conversation_type"] = conversation_type

        raw_text = self._generate(
            prompt,
            system_vars=system_vars,
            stream=True,
            stream_label=f"Generating conversation transcript ({language})…",
            **overrides,
        )
        transcript = self._clean(raw_text)

        from ..logger import Logger
        Logger.debug(f"Generator transcript:\n{transcript}")
        return transcript

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        title: str,
        context: str,
        language: str,
        conversation_type: str | None,
        agent_emotion: str | None,
        user_emotion: str | None,
        agent_accent: str | None,
        user_accent: str | None,
        gender_pair: str | None,
        previous_transcript: str | None,
        feedback: str | None,
        target_duration_sec: float | None = None,
        include_numbers: bool = False,
    ) -> str:
        """Assemble the user-side prompt sent alongside the Langfuse system prompt."""
        lines: list[str] = [
            "Write a realistic, natural-sounding multi-turn conversation as tagged "
            "plain text (one turn per line), following the format in the system prompt.",
            "",
            "## Topic",
            f"**Title:** {title}",
            f"**Context:** {context}",
            f"**Language:** {language}",
        ]

        lines.extend(_language_directive(language))

        if conversation_type:
            lines.append(f"**Conversation type:** {conversation_type}")
        if agent_emotion:
            lines.append(f"**Speaker 1 (agent) emotion:** {agent_emotion}")
        if user_emotion:
            lines.append(f"**Speaker 2 (user) emotion:** {user_emotion}")
        if agent_accent:
            lines.append(f"**Speaker 1 (agent) accent:** {agent_accent}")
        if user_accent:
            lines.append(f"**Speaker 2 (user) accent:** {user_accent}")
        if gender_pair:
            lines.append(
                f"**Gender pair (speaker_1-speaker_2, M=Male, F=Female):** {gender_pair}"
            )

        # Concrete, high-salience accent guidance (language-aware; no-op for Normal).
        lines.extend(_accent_guidance(agent_accent, user_accent, language))

        if target_duration_sec is not None:
            lines.append("")
            lines.append("## Target duration")
            lines.append(
                f"Aim for a conversation that lasts approximately "
                f"**{target_duration_sec:.0f} seconds** (~{target_duration_sec / 60:.1f} "
                f"minutes) when spoken at a natural pace (~2-3 words/second). Write "
                f"enough turns and content to fill it — do not end early."
            )

        # Number inclusion is decided per-conversation by the runner.
        lines.extend(_number_directive(include_numbers))

        if previous_transcript and feedback:
            lines.append("")
            lines.append("## PREVIOUS ATTEMPT & VALIDATION FEEDBACK")
            lines.append(
                "Your previous attempt was REJECTED for the issues below. Do NOT "
                "resubmit it with cosmetic word-swaps — actually FIX each issue: "
                "rewrite the flagged lines and any similar ones, replace wording that "
                "read as robotic or script-like, and add the missing accent/realism "
                "flavour. Keep the exact plain-text tag format, but the dialogue must "
                "be visibly different wherever the feedback points."
            )
            lines.append("")
            lines.append("### Feedback / issues to fix:")
            lines.append(feedback)
            lines.append("")
            lines.append("### Previous transcript:")
            lines.append(previous_transcript)

        lines.append("")
        lines.append(
            "Output ONLY the transcript lines — no JSON, no headings, no commentary."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Output cleaning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean(text: str) -> str:
        """Strip code fences / stray wrapping and keep only real dialogue lines.

        The generator is instructed to emit bare ``S1:``/``S2:`` lines, but small
        models sometimes wrap them in ```` ``` ```` fences or add a stray intro
        line. We drop fences and any line that doesn't start with a speaker tag,
        leaving a clean transcript for the formatter.
        """
        raw = (text or "").strip()
        if not raw:
            raise ValueError("Conversation generator returned an empty transcript.")

        kept: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("```"):
                continue
            # Keep only lines that look like a speaker turn: "S1:" / "S2 :" etc.
            head = stripped[:4].replace(" ", "").lower()
            if head.startswith("s1:") or head.startswith("s2:"):
                kept.append(stripped)

        if not kept:
            raise ValueError(
                "Conversation generator produced no parseable 'S1:'/'S2:' turns.\n"
                f"---\n{raw}"
            )
        return "\n".join(kept)
