# voice_collection

Fetch unique speakers from English (**Svarah**) and Hindi (**IndicVoices-R**) corpora, keep one best clip per speaker, and publish to the HuggingFace Storage Bucket `hf://buckets/inavlabs/voice_collection`.

## Selection rules

For each speaker (who may appear in many utterances):

1. If any instance is **≥ 10 s**, keep the **shortest** one that still meets 10 s (closest to the target without extra audio).
2. Else if any instance is **≥ 5 s**, keep the **longest** one below 10 s.
3. Else discard the speaker (all clips shorter than 5 s).

## Output layout

```
output/voice_collection/
  english/
    male/<speaker_id>/audio.wav
    male/<speaker_id>/metadata.json
    female/...
  hindi/
    male/...
  _manifests/
    english_summary.json
    hindi_summary.json
```

Uploaded to: `hf://buckets/inavlabs/voice_collection/{language}/{gender}/{speaker}/`

Set `"ANONYMIZE_SPEAKER_NAMES": true` in config to use `speaker_1`, `speaker_2`, … per gender instead of dataset speaker ids.

## Setup

```bash
pip install datasets huggingface_hub scipy soundfile numpy
cp voice_collection/config.json.example voice_collection/config.json
# Edit config.json: HF_TOKEN (write access), HF_BUCKET, dataset blocks
```

## Run

```bash
# Both languages, full pipeline (export + HF bucket upload)
python -m voice_collection.runner --language=all

# English only, local export without upload
python -m voice_collection.runner --language=english --dry-run

# Quick smoke test (5 speakers per language)
python -m voice_collection.runner --language=all --max-speakers=5 --dry-run
```

## Architecture

| Module | Role |
|--------|------|
| `runner.py` | CLI entry point |
| `pipeline.py` | Orchestrates stream → select → export → upload |
| `sources/` | HuggingFace dataset adapters (Strategy + Factory) |
| `processing/speaker_selector.py` | Per-speaker clip selection |
| `processing/audio_codec.py` | Decode, resample, write WAV |
| `exporter.py` | `{language}/{gender}/{speaker}/` folder writer |
| `storage/` | HuggingFace bucket or local copy (Abstract storage + Factory) |
| `configuration_reader.py` | `config.json` loader |

## metadata.json

Each speaker folder includes transcript, duration, selection tier, source dataset/index, and dataset-specific fields (age, location, SNR, etc.).
