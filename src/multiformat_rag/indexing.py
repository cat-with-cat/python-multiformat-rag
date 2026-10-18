"""Persistent LlamaIndex vector index and traceable retrieval results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, MetadataMode

from multiformat_rag.exceptions import IndexNotFoundError

MANIFEST_FILENAME = "rag_manifest.json"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk with its original source metadata."""

    text: str
    score: float | None
    metadata: dict[str, Any]

    @property
    def source_label(self) -> str:
        parts = [str(self.metadata.get("source_file", "未知文件"))]
        if self.metadata.get("page_number") is not None:
            parts.append(f"第 {self.metadata['page_number']} 页")
        if self.metadata.get("slide_number") is not None:
            parts.append(f"第 {self.metadata['slide_number']} 张幻灯片")
        if self.metadata.get("row_number") is not None:
            parts.append(f"第 {self.metadata['row_number']} 行")
        parts.append(f"分块 {self.metadata.get('chunk_index', '?')}")
        return "，".join(parts)


class PersistentRAGIndex:
    """Build, persist, reload, and query a LlamaIndex ``VectorStoreIndex``."""

    def __init__(
        self,
        index: VectorStoreIndex,
        *,
        persist_dir: Path,
        embedding_model_name: str,
        manifest: dict[str, Any],
    ) -> None:
        self._index = index
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name
        self.manifest = manifest

    @classmethod
    def build(
        cls,
        nodes: list[BaseNode],
        *,
        persist_dir: str | Path,
        embed_model: BaseEmbedding,
        embedding_model_name: str,
        allow_overwrite: bool = False,
        show_progress: bool = False,
    ) -> "PersistentRAGIndex":
        if not nodes:
            raise ValueError("没有可用于构建索引的非空分块。")

        target = Path(persist_dir).expanduser().resolve()
        if target.exists() and not target.is_dir():
            raise FileExistsError(f"索引路径不是目录：{target}")
        if target.exists() and any(target.iterdir()) and not allow_overwrite:
            raise FileExistsError(
                f"索引目录非空：{target}。请改用新目录或显式传入 --overwrite。"
            )
        target.mkdir(parents=True, exist_ok=True)

        index = VectorStoreIndex(
            nodes,
            embed_model=embed_model,
            show_progress=show_progress,
        )
        index.storage_context.persist(persist_dir=str(target))

        source_files = sorted(
            {
                str(node.metadata.get("source_file"))
                for node in nodes
                if node.metadata.get("source_file")
            }
        )
        manifest: dict[str, Any] = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": embedding_model_name,
            "chunk_count": len(nodes),
            "source_files": source_files,
        }
        (target / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cls(
            index,
            persist_dir=target,
            embedding_model_name=embedding_model_name,
            manifest=manifest,
        )

    @classmethod
    def load(
        cls,
        persist_dir: str | Path,
        *,
        embed_model: BaseEmbedding,
        expected_embedding_model: str | None = None,
    ) -> "PersistentRAGIndex":
        target = Path(persist_dir).expanduser().resolve()
        manifest = cls.read_manifest(target)
        stored_model = str(manifest["embedding_model"])
        if expected_embedding_model and expected_embedding_model != stored_model:
            raise ValueError(
                "加载索引时的 Embedding 模型与建库模型不一致："
                f"期望 {stored_model}，实际 {expected_embedding_model}。"
            )

        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(target))
            index = load_index_from_storage(
                storage_context,
                embed_model=embed_model,
            )
        except Exception as exc:
            raise IndexNotFoundError(
                f"索引文件无法加载或已经损坏：{target}"
            ) from exc
        return cls(
            index,
            persist_dir=target,
            embedding_model_name=stored_model,
            manifest=manifest,
        )

    @staticmethod
    def read_manifest(persist_dir: str | Path) -> dict[str, Any]:
        manifest_path = Path(persist_dir).expanduser().resolve() / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise IndexNotFoundError(
                f"未找到索引清单：{manifest_path}。请先执行 index 命令。"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise IndexNotFoundError(f"索引清单不可读取：{manifest_path}") from exc
        if manifest.get("format_version") != 1 or not manifest.get("embedding_model"):
            raise IndexNotFoundError(f"索引清单格式无效：{manifest_path}")
        return manifest

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("检索问题不能为空。")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是大于 0 的整数。")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity 必须位于 [-1, 1]。")

        retriever = self._index.as_retriever(similarity_top_k=top_k)
        node_results = retriever.retrieve(query)
        results: list[SearchResult] = []
        for item in node_results:
            score = float(item.score) if item.score is not None else None
            # LlamaIndex's in-memory vector store returns cosine similarity:
            # larger scores are better. Never invert it as an L2 distance.
            if min_similarity is not None and (score is None or score < min_similarity):
                continue
            results.append(
                SearchResult(
                    text=item.node.get_content(metadata_mode=MetadataMode.NONE),
                    score=score,
                    metadata=dict(item.node.metadata),
                )
            )
        return results
