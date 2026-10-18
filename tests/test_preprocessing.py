from llama_index.core import Document

from multiformat_rag.preprocessing import clean_documents, clean_text


def test_clean_text_normalizes_unicode_controls_and_whitespace() -> None:
    raw = "ＡＢＣ\x00\t  first  \r\n\r\n\r\n second\u200b"

    assert clean_text(raw) == "ABC first\n\nsecond"


def test_clean_documents_preserves_metadata_and_filters_empty() -> None:
    documents = [
        Document(text=" useful ", metadata={"source_id": "one"}),
        Document(text="\x00\n", metadata={"source_id": "two"}),
    ]

    cleaned = clean_documents(documents)

    assert len(cleaned) == 1
    assert cleaned[0].text == "useful"
    assert cleaned[0].metadata == {"source_id": "one"}

