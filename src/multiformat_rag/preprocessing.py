"""Text normalization that preserves useful paragraph boundaries."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from llama_index.core import Document

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize Unicode, control characters, and whitespace.

    Newlines are retained and runs of blank lines are collapsed to one blank
    line, keeping paragraph structure available to the chunker.
    """

    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"}
        or not unicodedata.category(character).startswith("C")
    )
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    normalized = _EXCESSIVE_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def clean_documents(documents: Iterable[Document]) -> list[Document]:
    """Clean documents, preserve metadata, and remove empty results."""

    cleaned: list[Document] = []
    for document in documents:
        text = clean_text(document.text)
        if not text:
            continue
        cleaned.append(Document(text=text, metadata=dict(document.metadata)))
    return cleaned
