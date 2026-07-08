"""Speaker-selection helpers for the voice-collection pipeline.

Keep this package lightweight at import time: selection-only tests should not
have to import optional audio dependencies such as ``scipy`` just to access
``SpeakerAudioSelector``.
"""

from .speaker_selector import SpeakerAudioSelector

__all__ = ["SpeakerAudioSelector"]
