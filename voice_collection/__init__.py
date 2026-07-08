"""voice_collection: fetch, filter and publish per-speaker voice samples.

Pulls unique speakers out of the Svarah (English) and IndicVoices-R (Hindi)
speech corpora, keeps a single best-duration clip per speaker, and publishes
the result to the ``inavlabs/voice_collection`` HuggingFace bucket in a
``{language}/{gender}/{speaker}`` layout.

See ``README.md`` for setup and ``runner.py`` for the CLI entry point.
"""
