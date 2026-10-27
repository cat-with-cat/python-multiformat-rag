"""Python Multi-format Document RAG."""

from multiformat_rag.config import AppConfig
from multiformat_rag.indexing import PersistentRAGIndex
from multiformat_rag.loaders import MultiFormatDocumentLoader
from multiformat_rag.service import RAGService

__all__ = [
    "AppConfig",
    "MultiFormatDocumentLoader",
    "PersistentRAGIndex",
    "RAGService",
]

__version__ = "0.3.0"
