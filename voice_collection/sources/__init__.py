"""Dataset-source abstraction for the voice-collection pipeline.

Only re-export the lightweight abstractions here so callers that do not need to
touch Hugging Face datasets can import the package without pulling in optional
third-party dependencies.
"""

from .base_source import BaseDatasetSource, DatasetLoadError
from .factory import DatasetSourceFactory

__all__ = ["BaseDatasetSource", "DatasetLoadError", "DatasetSourceFactory"]
