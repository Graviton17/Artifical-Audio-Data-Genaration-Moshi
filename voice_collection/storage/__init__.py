"""Storage abstraction for uploading the ``{language}/{gender}/{speaker}`` tree."""
from .base_storage import BaseStorage, StorageError
from .factory import create_storage
from .huggingface_storage import HuggingFaceStorage
from .local_storage import LocalStorage

__all__ = ["BaseStorage", "StorageError", "create_storage", "HuggingFaceStorage", "LocalStorage"]
