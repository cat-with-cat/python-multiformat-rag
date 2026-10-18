"""LlamaIndex SentenceSplitter integration with metadata guarantees."""

from __future__ import annotations

from collections.abc import Iterable

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size 必须是整数。")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int):
        raise ValueError("chunk_overlap 必须是整数。")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0。")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextNode]:
    """Split documents while retaining all source metadata on every node."""

    validate_chunk_parameters(chunk_size, chunk_overlap)
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[TextNode] = []
    for document in documents:
        source_metadata = dict(document.metadata)
        if not source_metadata.get("source_id"):
            raise ValueError("文档缺少 source_id，无法建立可追溯分块。")
        nodes = splitter.get_nodes_from_documents([document], show_progress=False)
        for chunk_index, node in enumerate(nodes):
            metadata = dict(source_metadata)
            metadata["chunk_index"] = chunk_index
            node.metadata = metadata
            node.id_ = f"{source_metadata['source_id']}:chunk:{chunk_index}"
            chunks.append(node)
    return chunks
