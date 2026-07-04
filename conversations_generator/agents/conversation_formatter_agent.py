"""Agent that formats a tagged plain-text transcript into schema JSON turns.

This is the second half of the two-stage generation pipeline. It consumes the
tagged plain text produced by
:class:`~conversations_generator.agents.conversation_generator_agent.ConversationGeneratorAgent`
and produces the full list of turn dicts following
``conversation_field_schema.json`` (turn_id, speaker, text, emotion, planned
timing, overlap/interruption metadata, etc.).

The LLM is prompted to emit the **full** ``conversation_field_schema.json`` shape
(all 14 fields per turn), but the work is still split so each half stays reliable:

* **LLM step** — the ``conversation-formatter-agent`` Langfuse prompt asks the
  model to convert the transcript into the full schema JSON. From that output we
  keep only the *semantic* fields it can read straight off the tags —
  ``speaker``, ``turn_type``, ``emotion``, ``text`` (plus ``overlaps_with`` as a
  hint to which speaker a relation points at, and ``join_ratio`` as a hint to
  *where* inside that partner turn a Backchanneling/Overlapping line begins —
  see below). This is a near-mechanical read of markup the generator already
  wrote, so even a small model is dependable, and it tolerates minor format drift.

* **Deterministic step** — plain Python (:meth:`_assemble`) then (re)assigns
  sequential ``turn_id``s, resolves each relation to a concrete partner turn (the
  *nearest preceding floor-holding turn by the other speaker*), fills
  ``overlaps_with`` / ``overlaps_kind`` / ``interrupted`` / ``interrupted_by``
  **symmetrically**, and lays out ``planned_start_sec`` / ``planned_end_sec`` so
  that every overlap / interruption / backchannel satisfies
  :class:`~conversations_generator.agents.conversation_validator_manual.ConversationValidatorManual`
  *by construction*. That is the whole point of the split: the numeric/relational
  consistency small LLMs kept getting wrong (timing, symmetric references) is
  produced by code, not trusted from the model — so the returned turns always
  pass manual validation.

Two things keep the generator's *intent* and the formatter's *timing* in sync,
without ever exposing internal machinery in the output schema:

1. **Join ratio.** The generator tags Backchanneling/Overlapping lines with
   ``-> S<ref>@<ratio>`` (see ``conversation-generator-agent.md``), a 0.0-1.0
   estimate of how far into the partner turn's utterance this line begins.
   The formatter LLM lifts that into a transient ``join_ratio`` field, which
   :meth:`_assemble` uses to place ``planned_start_sec`` at that exact point
   (instead of an arbitrary fixed offset), then discards — it is never part of
   a returned turn dict.
2. **Shared speech-rate model.** Both this file's :meth:`_estimate_duration`
   and the generator's pacing/length instructions use the *same*
   words/sec + chars/sec numbers (see the "Speech-rate model" constants below
   and the matching section in both prompt files), so a turn's estimated
   spoken duration doesn't drift between what the generator assumed while
   writing and what the formatter computes while laying out timing.

The system prompt is managed in Langfuse under the name
``conversation-formatter-agent``.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

# ------------------------------------------------------------------ #
# Enums / constants (mirror conversation_field_schema.json + manual validator)
# ------------------------------------------------------------------ #
VALID_SPEAKERS = {"speaker_1", "speaker_2"}
VALID_EMOTIONS = {"Neutral", "Happy", "Sad", "Angry"}
VALID_TURN_TYPES = {"Normal", "Overlapping", "Interruption", "Backchanneling"}

# Map the plain-text tags / loose LLM synonyms onto the schema turn_type enum.
_TURN_TYPE_ALIASES = {
    "normal": "Normal",
    "overlap": "Overlapping",
    "overlapping": "Overlapping",
    "interrupt": "Interruption",
    "interruption": "Interruption",
    "interrupted": "Interruption",
    "backchannel": "Backchanneling",
    "backchanneling": "Backchanneling",
    "back-channel": "Backchanneling",
}

# ------------------------------------------------------------------ #
# Speech-rate model — deterministic layout parameters (seconds)
#
# SHARED WITH THE GENERATOR: the same words/sec + chars/sec numbers are
# documented in the "Speech-rate model" section of both
# ``data/prompts/conversation-generator-agent.md`` (so the LLM paces turn
# count/length against a target duration using this rate) and
# ``data/prompts/conversation-formatter-agent.md`` (so the timing rules the
# formatter LLM is told to follow match what this code actually computes).
# If you change these constants, update both prompt files to match, or the
# generator's target-duration pacing and the formatter's actual timing will
# drift apart.
#
# Duration is estimated by blending a word-count estimate with a
# character-count estimate (rather than trusting either alone) since word
# count is a poor proxy when word length varies a lot (e.g. "70,000" or
# "graphics" vs "hi"), while raw character count is a poor proxy for very
# short utterances. AVG_CHARS_PER_WORD is the bridge constant that ties the
# two rates to one underlying assumption.
# ------------------------------------------------------------------ #
AVG_CHARS_PER_WORD = 5.2      # avg word length incl. trailing space (Hinglish/Hindi informal speech)
WORDS_PER_SEC = {             # natural conversational pace (Hinglish/Hindi), by emotion
    "Neutral": 2.5,
    "Happy": 2.5,
    "Sad": 2.0,               # sad speech is a bit slower
    "Angry": 3.0,             # angry speech is a bit faster
}
CHARS_PER_SEC = {emotion: round(wps * AVG_CHARS_PER_WORD, 1) for emotion, wps in WORDS_PER_SEC.items()}
MIN_TURN_SEC = 0.6            # floor on any single turn's duration
GAP_SEC = 0.3                # natural pause between sequential (Normal) turns
OVERLAP_OFFSET_SEC = 0.5     # fallback offset when a turn carries no join ratio
INTERRUPT_LEAD_SEC = 0.4     # how long before the victim's end the cut-in starts
INTERRUPT_EXTEND_SEC = 0.3   # how far past the victim's end the interrupter runs
BACKCHANNEL_MAX_FRAC = 0.8   # a backchannel fills at most this fraction of its host


class ConversationFormatterAgent(BaseAgent):
    """Convert a tagged plain-text transcript into schema-valid turn dicts.

    One LLM call parses the transcript into simple per-line fields; the rest is
    deterministic Python that produces the final ``conversation_field_schema.json``
    records with consistent timing and overlap/interruption metadata.
    """

    prompt_name = "conversation-formatter-agent"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        transcript: str,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        language: str | None = None,
        feedback: str | None = None,
        previous_output: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> list[dict[str, Any]]:
        """Format ``transcript`` into a list of schema turn dicts.

        Parameters
        ----------
        transcript : str
            The tagged plain-text conversation from
            :class:`ConversationGeneratorAgent`.
        agent_emotion, user_emotion : str | None
            Dominant emotion per speaker, used as a fallback when a line's
            emotion is missing/invalid.
        language : str | None
            Currently unused by the timing model; accepted for forward
            compatibility / logging.
        feedback : str | None
            Manual-validation feedback from a previous formatting attempt of the
            *same* transcript. Used on formatter retries so the model can fix the
            formatting without the transcript changing.
        previous_output : list[dict] | None
            The previous attempt's formatted turns, shown alongside ``feedback``
            so the model can see and correct what it produced.
        **overrides
            Extra kwargs forwarded to the LLM (temperature, max_tokens, etc.).

        Returns
        -------
        list[dict]
            Ordered turn dicts matching ``conversation_field_schema.json``.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Formatter received an empty transcript.")

        prompt = self._build_prompt(transcript, feedback=feedback, previous_output=previous_output)

        # Parsing should be near-deterministic, not creative.
        overrides.setdefault("temperature", 0.1)
        overrides.setdefault("response_format", {"type": "json_object"})
        raw_result = self._generate_json(
            prompt,
            stream=True,
            stream_label="Formatting transcript into schema turns…",
            **overrides,
        )

        from ..logger import Logger
        Logger.debug(f"Formatter LLM Output:\n{json.dumps(raw_result, indent=2, ensure_ascii=False)}")

        parsed = self._extract_lines(raw_result)
        turns = self._assemble(parsed, agent_emotion=agent_emotion, user_emotion=user_emotion)
        return turns

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        transcript: str,
        feedback: str | None = None,
        previous_output: list[dict[str, Any]] | None = None,
    ) -> str:
        """Assemble the user-side prompt sent alongside the Langfuse system prompt."""
        lines: list[str] = [
            "Convert the plain-text conversation transcript below into the full "
            "schema JSON described in the system prompt: one turn object per "
            "dialogue line, in order, with all 14 schema fields plus the internal "
            "join_ratio field described in the system prompt.",
            "",
            "## Transcript",
            transcript,
        ]

        if feedback and previous_output:
            lines += [
                "",
                "## PREVIOUS FORMATTING ATTEMPT & VALIDATION FEEDBACK",
                "Your previous formatting of THIS SAME transcript failed manual "
                "validation. Fix the issues below and re-format the transcript. Do "
                "not change the dialogue — only correct the schema/structure.",
                "",
                "### Feedback / errors to fix:",
                feedback,
                "",
                "### Your previous formatted output (JSON):",
                "```json",
                json.dumps({"turns": previous_output}, ensure_ascii=False, indent=2),
                "```",
            ]

        lines += [
            "",
            "Return ONLY the single JSON object — no prose, no markdown fences.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # LLM output extraction (keep the semantic fields; rebuild the rest)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_lines(result: Any) -> list[dict[str, Any]]:
        """Reduce the model's full-schema output to the fields code trusts.

        The LLM emits the full ``conversation_field_schema.json`` shape, but only
        the semantic fields it can read straight off the tags are kept —
        ``{speaker, turn_type, emotion, text}`` — plus ``ref_speaker`` (derived
        from the LLM's ``overlaps_with`` hint, i.e. the speaker of the turn a
        relation points at). Everything relational/numeric is rebuilt by
        :meth:`_assemble`. Accepts ``{"turns": [...]}`` or a bare list; turns with
        no usable text are dropped.
        """
        if isinstance(result, dict):
            if "turns" in result:
                items = result["turns"]
            else:
                values = list(result.values())
                items = values[0] if len(values) == 1 and isinstance(values[0], list) else None
                if items is None:
                    raise ValueError(
                        f"Formatter expected a 'turns' list, got keys: {list(result.keys())}"
                    )
        elif isinstance(result, list):
            items = result
        else:
            raise ValueError(f"Formatter expected a list/dict, got {type(result).__name__}")

        # First pass: map each turn_id the LLM emitted to its speaker, so a
        # relation's ``overlaps_with`` can be resolved to "which speaker".
        id_to_speaker: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            low = {str(k).lower(): v for k, v in item.items()}
            tid = low.get("turn_id")
            sp = str(low.get("speaker", "")).strip().lower()
            if tid is not None and sp in VALID_SPEAKERS:
                id_to_speaker[str(tid)] = sp

        parsed: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item = {str(k).lower(): v for k, v in item.items()}

            text = str(item.get("text", "")).strip()
            if not text:
                continue

            speaker = str(item.get("speaker", "")).strip().lower()
            if speaker not in VALID_SPEAKERS:
                # Tolerate "s1"/"1"/"speaker1" style values.
                digits = "".join(ch for ch in speaker if ch.isdigit())
                speaker = f"speaker_{digits}" if digits in {"1", "2"} else "speaker_1"

            turn_type = _TURN_TYPE_ALIASES.get(
                str(item.get("turn_type", "")).strip().lower(), "Normal"
            )

            # Prefer an explicit ref_speaker; otherwise infer it from the speaker
            # of the turn this one overlaps with. Left as None when neither is
            # usable — _assemble then defaults to "the other speaker".
            ref_speaker = item.get("ref_speaker")
            if ref_speaker is None:
                ref_speaker = id_to_speaker.get(str(item.get("overlaps_with")))
            if ref_speaker is not None:
                ref = str(ref_speaker).strip().lower()
                if ref not in VALID_SPEAKERS:
                    digits = "".join(ch for ch in ref if ch.isdigit())
                    ref = f"speaker_{digits}" if digits in {"1", "2"} else None
                ref_speaker = ref

            emotion = str(item.get("emotion", "")).strip().capitalize()
            if emotion not in VALID_EMOTIONS:
                emotion = None  # resolved later against speaker default

            # Transient join-point hint: for [backchannel]/[overlap] lines the
            # generator writes "-> S<ref>@<ratio>" (see conversation-generator-
            # agent.md), and the formatter LLM is asked to lift that <ratio>
            # into a "join_ratio" field. It is never part of the output schema —
            # only used internally to place planned_start_sec precisely (see
            # _assemble) — so it's parsed here the same tolerant way as
            # ref_speaker and dropped again before turns are returned.
            ref_ratio = item.get("join_ratio", item.get("ratio"))
            try:
                ref_ratio = float(ref_ratio) if ref_ratio is not None else None
            except (TypeError, ValueError):
                ref_ratio = None
            if ref_ratio is not None:
                ref_ratio = max(0.0, min(1.0, ref_ratio))

            parsed.append(
                {
                    "speaker": speaker,
                    "turn_type": turn_type,
                    "ref_speaker": ref_speaker,
                    "ref_ratio": ref_ratio,
                    "emotion": emotion,
                    "text": text,
                }
            )

        if not parsed:
            raise ValueError("Formatter parsed zero usable turns from the transcript.")
        return parsed

    # ------------------------------------------------------------------ #
    # Deterministic assembly: ids, references, timing
    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        parsed: list[dict[str, Any]],
        *,
        agent_emotion: str | None,
        user_emotion: str | None,
    ) -> list[dict[str, Any]]:
        """Build full schema turns with consistent references + timing.

        Single forward pass. Each relational turn is bound to the nearest
        preceding *floor-holding* (non-backchannel), not-yet-claimed turn by its
        referenced speaker; if none is available the turn is downgraded to Normal.
        Timing is laid out to satisfy the manual validator's overlap/interruption/
        backchannel conditions with margin.
        """
        default_emotion = {
            "speaker_1": self._clamp_emotion(agent_emotion),
            "speaker_2": self._clamp_emotion(user_emotion),
        }

        turns: list[dict[str, Any]] = []
        for i, p in enumerate(parsed, start=1):
            emotion = p["emotion"] or default_emotion.get(p["speaker"], "Neutral")
            turns.append(
                {
                    "turn_id": f"t{i}",
                    "speaker": p["speaker"],
                    "text": p["text"],
                    "emotion": emotion,
                    "planned_start_sec": None,
                    "planned_end_sec": None,
                    "real_start_sec": None,
                    "real_end_sec": None,
                    "error_time": None,
                    "turn_type": p["turn_type"],
                    "overlaps_with": None,
                    "overlaps_kind": None,
                    "interrupted": False,
                    "interrupted_by": None,
                    # transient hints, stripped before returning
                    "_ref_speaker": p["ref_speaker"],
                    "_ref_ratio": p["ref_ratio"],
                }
            )

        floor_end = 0.0          # end time of the last floor-holding turn
        claimed: set[str] = set()  # turn_ids already bound into a relationship

        for idx, t in enumerate(turns):
            ttype = t["turn_type"]
            dur = self._estimate_duration(t["text"], t["emotion"])

            partner = None
            if ttype != "Normal":
                partner = self._find_partner(turns, idx, t["_ref_speaker"], claimed)
                if partner is None:
                    # No valid partner -> behave as a plain Normal turn.
                    ttype = t["turn_type"] = "Normal"

            if ttype == "Normal":
                start = floor_end + GAP_SEC if floor_end > 0 else 0.0
                end = start + dur
                floor_end = end

            elif ttype == "Overlapping":
                assert partner is not None  # guaranteed: non-Normal with a resolved partner
                p_s, p_e = partner["planned_start_sec"], partner["planned_end_sec"]
                span = p_e - p_s
                ratio = t["_ref_ratio"]
                # Prefer the generator's own join-point ("-> S<ref>@<ratio>") over
                # the fixed fallback offset, so the collision lands exactly where
                # the dialogue actually implies it should.
                raw_offset = ratio * span if ratio is not None else min(OVERLAP_OFFSET_SEC, span * 0.5)
                # The collision MUST start strictly inside the partner's span: the
                # manual validator requires first_start < second_start < first_end.
                # A ratio of 0.0 (which the generator uses for near-simultaneous
                # overlaps) would put the start exactly at the partner's start and
                # fail that check — clamp to a small positive minimum, and keep it
                # safely before the partner's end.
                lo = min(0.15, span * 0.5)
                offset = min(max(raw_offset, lo), max(lo, span - 0.1))
                start = p_s + offset
                end = start + dur
                self._link(t, partner, "Overlapping", claimed)
                floor_end = max(p_e, end)

            elif ttype == "Interruption":
                # This turn is the interrupter; the partner is the victim.
                assert partner is not None  # guaranteed: non-Normal with a resolved partner
                v_s, v_e = partner["planned_start_sec"], partner["planned_end_sec"]
                start = min(v_e - INTERRUPT_LEAD_SEC, v_e - 0.05)
                start = max(start, v_s + 0.05)
                end = start + dur
                if end <= v_e:
                    end = v_e + INTERRUPT_EXTEND_SEC  # extend past victim's end
                self._link(t, partner, "Interruption", claimed)
                partner["interrupted"] = True
                partner["interrupted_by"] = t["turn_id"]
                floor_end = end

            else:  # Backchanneling — nested inside the host, floor unchanged
                assert partner is not None  # guaranteed: non-Normal with a resolved partner
                h_s, h_e = partner["planned_start_sec"], partner["planned_end_sec"]
                span = h_e - h_s
                sub_dur = max(0.1, min(dur, span * BACKCHANNEL_MAX_FRAC))
                ratio = t["_ref_ratio"]
                if ratio is not None:
                    # Land the backchannel at the word-position the generator
                    # intended.
                    start = h_s + ratio * span
                else:
                    start = h_s + (span - sub_dur) / 2  # fallback: centered
                # Clamp fully inside the host AND strictly after its start: a
                # backchannel at ratio 0.0 would otherwise begin exactly at the
                # host's start, and the validator's "earlier turn is the host"
                # tie-break would then mistake the short backchannel for the host.
                start = min(max(start, h_s + 0.1), h_e - sub_dur)
                end = start + sub_dur
                self._link(t, partner, "Backchanneling", claimed)
                # host keeps the floor: do NOT advance floor_end, do NOT interrupt it

            t["planned_start_sec"] = round(start, 2)
            t["planned_end_sec"] = round(end, 2)

        for t in turns:
            t.pop("_ref_speaker", None)
            t.pop("_ref_ratio", None)
        return turns

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_partner(
        turns: list[dict[str, Any]],
        idx: int,
        ref_speaker: str | None,
        claimed: set[str],
    ) -> dict[str, Any] | None:
        """The immediately preceding floor-holding turn, if it's a usable partner.

        A relationship is always with the turn currently holding the floor — the
        line "next to me" — so we look back only to the nearest non-backchannel
        turn. It is a valid partner only if it belongs to the referenced (other)
        speaker and isn't already bound into another relationship. Otherwise we
        return ``None`` and the caller downgrades this turn to Normal, rather than
        reaching further back (which would place it out of time order).

        The parsed reference is coerced to "the other speaker" when it's missing
        or points at this turn's own speaker, since a relationship is never with
        the same speaker.
        """
        own_speaker = turns[idx]["speaker"]
        if ref_speaker not in VALID_SPEAKERS or ref_speaker == own_speaker:
            ref_speaker = "speaker_2" if own_speaker == "speaker_1" else "speaker_1"

        for j in range(idx - 1, -1, -1):
            cand = turns[j]
            if cand["turn_type"] == "Backchanneling":
                continue  # backchannels don't hold the floor; look past them
            # First floor-holding turn found: usable only if it's the referenced
            # speaker and still free. If not, don't reach back any further.
            if cand["speaker"] == ref_speaker and cand["turn_id"] not in claimed:
                return cand
            return None
        return None

    @staticmethod
    def _link(turn: dict[str, Any], partner: dict[str, Any], kind: str, claimed: set[str]) -> None:
        """Set the symmetric overlaps_with/overlaps_kind on both turns and claim them."""
        turn["overlaps_with"] = partner["turn_id"]
        turn["overlaps_kind"] = kind
        partner["overlaps_with"] = turn["turn_id"]
        partner["overlaps_kind"] = kind
        claimed.add(turn["turn_id"])
        claimed.add(partner["turn_id"])

    @staticmethod
    def _estimate_duration(text: str, emotion: str) -> float:
        """Estimate a turn's spoken duration, blending a word-rate and char-rate estimate.

        Using both (rather than either alone) keeps this in sync with the same
        speech-rate model documented in the generator's system prompt: word
        count paces well for normal sentences but is a poor proxy for
        outliers (numbers, long compound words, one-word backchannels), so the
        two estimates are averaged for a steadier result.
        """
        words = max(1, len(text.split()))
        chars = max(1, len(text))
        wps = WORDS_PER_SEC.get(emotion, WORDS_PER_SEC["Neutral"])
        cps = CHARS_PER_SEC.get(emotion, CHARS_PER_SEC["Neutral"])
        by_words = words / wps
        by_chars = chars / cps
        return max(MIN_TURN_SEC, (by_words + by_chars) / 2)

    @staticmethod
    def _clamp_emotion(value: str | None) -> str:
        """Coerce an arbitrary emotion string to the schema enum (default Neutral)."""
        if value:
            cap = str(value).strip().capitalize()
            if cap in VALID_EMOTIONS:
                return cap
        return "Neutral"
