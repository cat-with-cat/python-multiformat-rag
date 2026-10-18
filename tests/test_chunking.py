import pytest
from llama_index.core import Document

from multiformat_rag.chunking import split_documents, validate_chunk_parameters


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11), (True, 0), (10, False)],
)
def test_rejects_invalid_chunk_parameters(chunk_size, chunk_overlap) -> None:
    with pytest.raises(ValueError):
        validate_chunk_parameters(chunk_size, chunk_overlap)


def test_sentence_splitter_creates_multiple_traceable_chunks() -> None:
    text = " ".join(f"Sentence {number} contains useful retrieval content." for number in range(80))
    document = Document(
        text=text,
        metadata={
            "source_file": "guide.txt",
            "file_type": "txt",
            "source_id": "source-001",
        },
    )

    chunks = split_documents([document], chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(chunk.metadata["source_file"] == "guide.txt" for chunk in chunks)
    assert all(chunk.metadata["source_id"] == "source-001" for chunk in chunks)
    assert all(chunk.id_.startswith("source-001:chunk:") for chunk in chunks)


def test_chunking_requires_source_id() -> None:
    with pytest.raises(ValueError, match="source_id"):
        split_documents(
            [Document(text="content", metadata={"source_file": "missing.txt"})],
            chunk_size=64,
            chunk_overlap=8,
        )

