"""Project-specific exceptions with user-facing messages."""


class RAGError(Exception):
    """Base exception for expected application errors."""


class ConfigurationError(RAGError):
    """Raised when required configuration is missing or invalid."""


class DocumentLoadError(RAGError):
    """Raised when an input document cannot be parsed."""


class UnsupportedFormatError(DocumentLoadError):
    """Raised when a file extension is not supported."""


class IndexNotFoundError(RAGError):
    """Raised when a persisted index is missing or incomplete."""
