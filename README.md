# Artificial Audio Data Generation for Moshi

Generate **synthetic two-speaker conversational audio** (with overlapping speech)
for training Moshi in low-resource languages (Gujarati, Hindi, …) where real
podcast data is unavailable. It is the **inverse** of `../Data-Processing-Moshi`
and produces the **same output contract**, so synthetic and real data are
interchangeable for `moshi-finetune`.

```
written conversation (JSON)  ──►  TTS  ──►  forced alignment  ──►  schedule (overlaps)
                                                                          │
   stereo wav (L=agent, R=other) + alignments json + dataset.jsonl  ◄─────┘
```

## Why this design

- **Words are never guessed.** You supply the text; we only compute *timing* via
  forced alignment of the known text against the synthesized audio. No ASR drift.
- **Channels are clean by construction.** Each speaker is synthesized and placed
  on its own mono track, then muxed to stereo. Overlap = simultaneous speech on
  *separate channels*, exactly what Moshi full-duplex needs — zero crosstalk.
- **The TTS model is swappable.** Everything talks to `BaseTTS`; the concrete
  model is chosen in one place (`factory.py`) from `config.yaml`.

## Install

```bash
pip install -r requirements.txt        # or: pip install -e ".[ml]"
```

## Run

```bash
python -m datagen.cli --input conversations --config config.yaml -v
# rebuild only the manifest from an existing output dir:
python -m datagen.cli --manifest-only --out dataset
```

Outputs land in `dataset/`:

```
dataset/
├── current_affairs_gu_01__user1_agent.wav   # L = user1 (agent), R = user2
├── current_affairs_gu_01__user1_agent.json
├── current_affairs_gu_01__user2_agent.wav   # channel-swapped variant
├── current_affairs_gu_01__user2_agent.json
├── dataset.jsonl                            # {"path": ..., "duration": ...}
└── report.json
```

## Input format (explicit overlap)

```jsonc
{
  "conversation_id": "current_affairs_gu_01",
  "language": "gu",
  "speakers": {
    "user1": { "voice": "<speaker id / style description / ref-audio path>" },
    "user2": { "voice": "..." }
  },
  "turns": [
    { "speaker": "user1", "text": "…" },
    { "speaker": "user2", "text": "…", "gap": 0.3 },     // 0.3s silence after prev
    { "speaker": "user1", "text": "…", "overlap": 0.4 }  // starts 0.4s BEFORE prev ends
  ]
}
```

`gap` and `overlap` are mutually exclusive per turn; `overlap` is clamped to
`schedule.max_overlap_sec`. `voice` is model-specific (a style prompt for
description-conditioned models, a speaker id, or a reference-audio path).

## Output JSON (per variant) — identical to the real pipeline

```jsonc
{
  "alignments": [["word", [start, end], "SPEAKER_MAIN"], …],  // read by the interleaver
  "segments": [{"speaker": "user1", "start": …, "end": …}, …],
  "transcript_by_speaker": { "SPEAKER_MAIN": [...], "SPEAKER_OTHER": [...] },
  "speakers": { "main": "user1", "user": "user2" },
  "purity": { "main_crosstalk": 0.0, "user_crosstalk": 0.0, "pass": true, "quarantined": false },
  "synthetic": { "source": "datagen", "tts_model": "…", "language": "gu" }
}
```

## Architecture (mirrors `Data-Processing-Moshi`)

| Concern | Where | Pattern |
|---|---|---|
| Typed config from YAML | `config.py` | nested dataclasses |
| Data passed between stages | `models.py` | DTOs (`GenContext`, `SynthClip`, `StreamVariant`, `Word`) |
| Stage contract | `stages/base.py` | Template Method + Chain of Responsibility |
| Orchestration | `pipeline.py` | Composite |
| Swappable algorithms | `strategies/` | Strategy (`BaseTTS`, `BaseAligner`, `BaseScheduler`) |
| Wiring | `factory.py` | Factory + registry |

**Pipeline:** `LoadScript → Synthesize → Align → Schedule → Streamize → BuildAlignments → WriteOutputs`.

### Swapping the TTS model

The first implementation is `IndicMioTTS` wired to `SPRINGLab/Indic-Mio`
(`tts.model_id` in `config.yaml`). If that id does not resolve on Hugging Face,
change `tts.model_id` to a verified one — no code change needed:

- `ai4bharat/indic-parler-tts` — multilingual (gu, hi, + ~20 Indic langs)
- `ai4bharat/IndicF5` — multilingual F5, 24 kHz, voice cloning
- `SPRINGLab/F5-Hindi-24KHz` — SPRINGLab F5, Hindi only

To add a genuinely different architecture, subclass `BaseTTS` in
`strategies/tts.py` and register it in `TTS_REGISTRY` (`factory.py`).

## Notes

- Output sample rate is 24 kHz (Mimi). Keep `tts.target_sr == sample_rate` to
  avoid resampling.
- If `ctc-forced-aligner` is unavailable, set `aligner.name: heuristic` (or rely
  on `fallback_heuristic: true`) for deterministic char-proportional timing.





<!-- IMP  -->

python -m conversations_generator.runner --language=hindi
# generation=sarvam (forced), validation/formatter=gemma

python -m conversations_generator.runner --language=english --model=gemini
# generation=gemini, validation/formatter=gemma

python -m conversations_generator.runner --model=sarvam --validation=gemini
# generation=sarvam, validation/formatter=gemini
python -m conversations_generator.runner --language=hindi
python -m conversations_generator.runner --language=hinglish --model=gemini
python -m conversations_generator.runner --language=english