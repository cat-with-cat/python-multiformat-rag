from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding

from multiformat_rag.chunking import split_documents
from multiformat_rag.exceptions import IndexNotFoundError
from multiformat_rag.indexing import PersistentRAGIndex


def _nodes():
    documents = [
        Document(
            text="Alpha supports semantic document search.",
            metadata={
                "source_file": "products.csv",
                "file_type": "csv",
                "source_id": "csv-row-2",
                "row_number": 2,
            },
        ),
        Document(
            text="The manual explains persistent indexes.",
            metadata={
                "source_file": "manual.pdf",
                "file_type": "pdf",
                "source_id": "pdf-page-3",
                "page_number": 3,
            },
        ),
    ]
    return split_documents(documents, chunk_size=128, chunk_overlap=16)


def test_build_reload_and_retrieve_preserves_sources(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    embed_model = MockEmbedding(embed_dim=16)
    built = PersistentRAGIndex.build(
        _nodes(),
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name="mock-16",
    )

    assert built.manifest["chunk_count"] == 2
    assert (persist_dir / "rag_manifest.json").is_file()

    loaded = PersistentRAGIndex.load(
        persist_dir,
        embed_model=embed_model,
        expected_embedding_model="mock-16",
    )
    results = loaded.retrieve("persistent index", top_k=2, min_similarity=-1.0)

    assert len(results) == 2
    for result in results:
        assert result.score is not None
        assert result.metadata["source_file"]
        assert result.metadata["file_type"] in {"csv", "pdf"}
        assert result.metadata["source_id"]
        assert result.metadata["chunk_index"] == 0
    assert any(result.metadata.get("page_number") == 3 for result in results)
    assert any(result.metadata.get("row_number") == 2 for result in results)


def test_reload_rejects_different_embedding_model(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    embed_model = MockEmbedding(embed_dim=8)
    PersistentRAGIndex.build(
        _nodes(),
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name="mock-8",
    )

    with pytest.raises(ValueError, match="不一致"):
        PersistentRAGIndex.load(
            persist_dir,
            embed_model=embed_model,
            expected_embedding_model="another-model",
        )


def test_explicit_overwrite_replaces_old_index_without_pollution(
    tmp_path: Path,
) -> None:
    persist_dir = tmp_path / "index"
    embed_model = MockEmbedding(embed_dim=8)
    PersistentRAGIndex.build(
        _nodes(),
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name="mock-8",
    )
    replacement = Document(
        text="Only the replacement document should remain.",
        metadata={
            "source_file": "replacement.txt",
            "file_type": "txt",
            "source_id": "replacement-document",
        },
    )
    replacement_nodes = split_documents(
        [replacement], chunk_size=128, chunk_overlap=16
    )

    rebuilt = PersistentRAGIndex.build(
        replacement_nodes,
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name="mock-8",
        allow_overwrite=True,
    )
    loaded = PersistentRAGIndex.load(
        persist_dir,
        embed_model=embed_model,
        expected_embedding_model="mock-8",
    )
    results = loaded.retrieve("replacement", top_k=10, min_similarity=-1.0)

    assert rebuilt.manifest["chunk_count"] == 1
    assert rebuilt.manifest["source_files"] == ["replacement.txt"]
    assert len(results) == 1
    assert results[0].metadata["source_file"] == "replacement.txt"


def test_reload_reports_corrupted_index_cleanly(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    embed_model = MockEmbedding(embed_dim=8)
    PersistentRAGIndex.build(
        _nodes(),
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name="mock-8",
    )
    (persist_dir / "default__vector_store.json").write_text(
        "not valid json", encoding="utf-8"
    )

    with pytest.raises(IndexNotFoundError, match="损坏"):
        PersistentRAGIndex.load(
            persist_dir,
            embed_model=embed_model,
            expected_embedding_model="mock-8",
        )


def test_index_and_retrieval_validate_errors(tmp_path: Path) -> None:
    embed_model = MockEmbedding(embed_dim=8)
    with pytest.raises(ValueError, match="非空分块"):
        PersistentRAGIndex.build(
            [],
            persist_dir=tmp_path / "empty-index",
            embed_model=embed_model,
            embedding_model_name="mock-8",
        )

    index = PersistentRAGIndex.build(
        _nodes(),
        persist_dir=tmp_path / "index",
        embed_model=embed_model,
        embedding_model_name="mock-8",
    )
    with pytest.raises(ValueError, match="不能为空"):
        index.retrieve(" ")
    with pytest.raises(ValueError, match="top_k"):
        index.retrieve("query", top_k=0)
    with pytest.raises(ValueError, match="min_similarity"):
        index.retrieve("query", min_similarity=1.1)
