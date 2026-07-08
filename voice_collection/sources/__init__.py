"""Dataset-source abstraction for the voice-collection pipeline."""
from .base_source import BaseDatasetSource, DatasetLoadError
from .factory import DatasetSourceFactory
from .indicvoices_source import IndicVoicesDatasetSource
from .svarah_source import SvarahDatasetSource

__all__ = [
    "BaseDatasetSource",
    "DatasetLoadError",
    "DatasetSourceFactory",
    "IndicVoicesDatasetSource",
    "SvarahDatasetSource",
]
