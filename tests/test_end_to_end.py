from pathlib import Path

from llama_index.core.embeddings import MockEmbedding

from multiformat_rag.chunking import split_documents
from multiformat_rag.indexing import PersistentRAGIndex
from multiformat_rag.loaders import MultiFormatDocumentLoader
from multiformat_rag.preprocessing import clean_documents


def test_four_formats_build_reload_and_retrieve_with_metadata(
    synthetic_documents: Path,
    tmp_path: Path,
) -> None:
    documents = clean_documents(
        MultiFormatDocumentLoader().load(synthetic_documents)
    )
    nodes = split_documents(documents, chunk_size=128, chunk_overlap=16)
    embed_model = MockEmbedding(embed_dim=16)
    persist_dir = tmp_path / "four-format-index"

    built = PersistentRAGIndex.build(
        nodes,
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name="mock-16",
    )
    loaded = PersistentRAGIndex.load(
        persist_dir,
        embed_model=embed_model,
        expected_embedding_model="mock-16",
    )
    results = loaded.retrieve(
        "traceable document retrieval",
        top_k=len(nodes),
        min_similarity=-1.0,
    )

    assert set(built.manifest["source_files"]) == {
        "guide.txt",
        "manual.pdf",
        "products.csv",
        "slides.pptx",
    }
    assert len(results) == len(nodes)
    assert {result.metadata["file_type"] for result in results} == {
        "txt",
        "csv",
        "pdf",
        "pptx",
    }
    assert all(result.metadata["source_id"] for result in results)
    assert all("chunk_index" in result.metadata for result in results)
    assert any("page_number" in result.metadata for result in results)
    assert any("slide_number" in result.metadata for result in results)
    assert any("row_number" in result.metadata for result in results)
