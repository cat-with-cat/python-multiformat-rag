from pathlib import Path

import pytest

from multiformat_rag.exceptions import DocumentLoadError, UnsupportedFormatError
from multiformat_rag.loaders import MultiFormatDocumentLoader
from multiformat_rag.preprocessing import clean_documents


def test_loads_all_four_formats_with_granular_metadata(
    synthetic_documents: Path,
) -> None:
    documents = MultiFormatDocumentLoader().load(synthetic_documents)

    assert len(documents) == 7
    by_type = {
        file_type: [doc for doc in documents if doc.metadata["file_type"] == file_type]
        for file_type in {"txt", "csv", "pdf", "pptx"}
    }
    assert len(by_type["txt"]) == 1
    assert [doc.metadata["row_number"] for doc in by_type["csv"]] == [2, 3]
    assert [doc.metadata["page_number"] for doc in by_type["pdf"]] == [1, 2]
    assert [doc.metadata["slide_number"] for doc in by_type["pptx"]] == [1, 2]

    for document in documents:
        assert document.metadata["source_file"]
        assert not Path(document.metadata["source_file"]).is_absolute()
        assert len(document.metadata["source_id"]) == 20


def test_pdf_and_pptx_text_is_extractable(synthetic_documents: Path) -> None:
    loader = MultiFormatDocumentLoader()
    pdf_documents = loader.load(synthetic_documents / "manual.pdf")
    pptx_documents = loader.load(synthetic_documents / "slides.pptx")

    assert "indexing" in pdf_documents[0].text
    assert "persistence" in pdf_documents[1].text
    assert "document parsing" in pptx_documents[0].text
    assert "grounded answers" in pptx_documents[1].text


def test_empty_document_is_removed_after_cleaning(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("\x00  \n\n", encoding="utf-8")

    raw_documents = MultiFormatDocumentLoader().load(empty_file)
    assert len(raw_documents) == 1
    assert clean_documents(raw_documents) == []


def test_rejects_unsupported_and_missing_paths(tmp_path: Path) -> None:
    unsupported = tmp_path / "notes.docx"
    unsupported.write_bytes(b"not a real document")

    with pytest.raises(UnsupportedFormatError):
        MultiFormatDocumentLoader().load(unsupported)
    with pytest.raises(DocumentLoadError):
        MultiFormatDocumentLoader().load(tmp_path / "missing")

