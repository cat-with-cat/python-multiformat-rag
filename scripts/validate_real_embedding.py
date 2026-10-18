"""Run a local, CPU-only Chinese retrieval validation.

This script creates synthetic documents and validation artifacts below the
chosen output directory. It does not call an LLM or read API credentials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from multiformat_rag.chunking import split_documents
from multiformat_rag.config import DEFAULT_EMBEDDING_MODEL
from multiformat_rag.indexing import PersistentRAGIndex
from multiformat_rag.loaders import MultiFormatDocumentLoader
from multiformat_rag.preprocessing import clean_documents


SYNTHETIC_DOCUMENTS = {
    "rag_guide.txt": (
        "可追溯检索说明\n\n"
        "系统在解析文档时保存文件名和来源编号。文本分块后继续保留来源元数据，"
        "语义检索结果会展示文件名、分块编号以及 PDF 页码等定位信息。"
    ),
    "city_guide.txt": (
        "城市出行说明\n\n"
        "西安春季气候温和。参观博物馆前应查看开放时间，并优先选择公共交通。"
    ),
    "recipe.txt": (
        "家常菜说明\n\n"
        "制作番茄炒蛋时，先将鸡蛋炒熟盛出，再炒番茄，最后混合并调味。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".validation/real-embedding"),
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "documents"
    index_dir = output_dir / "index"
    data_dir.mkdir(parents=True, exist_ok=True)
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError(
            f"验证索引目录非空：{index_dir}。请指定新的 --output-dir。"
        )

    for filename, text in SYNTHETIC_DOCUMENTS.items():
        (data_dir / filename).write_text(text, encoding="utf-8")

    embed_model = HuggingFaceEmbedding(
        model_name=args.model,
        device="cpu",
        normalize=True,
    )
    probe_vector = embed_model.get_text_embedding("中文语义检索")
    documents = clean_documents(MultiFormatDocumentLoader().load(data_dir))
    nodes = split_documents(documents, chunk_size=128, chunk_overlap=16)
    built = PersistentRAGIndex.build(
        nodes,
        persist_dir=index_dir,
        embed_model=embed_model,
        embedding_model_name=args.model,
    )
    loaded = PersistentRAGIndex.load(
        index_dir,
        embed_model=embed_model,
        expected_embedding_model=args.model,
    )
    query = "怎样让语义检索结果能够追踪到原始文档的位置？"
    results = loaded.retrieve(query, top_k=3, min_similarity=None)
    scores = [result.score for result in results]
    scores_descending = all(
        left is not None and right is not None and left >= right
        for left, right in zip(scores, scores[1:])
    )
    top_result_expected = bool(
        results and results[0].metadata.get("source_file") == "rag_guide.txt"
    )
    required_metadata = {"source_file", "file_type", "source_id", "chunk_index"}
    metadata_complete = all(
        required_metadata.issubset(result.metadata) for result in results
    )
    if len(results) != 3 or not (
        scores_descending and top_result_expected and metadata_complete
    ):
        raise RuntimeError(
            "真实 Embedding 验证失败：Top-K 排序、预期首条结果或来源元数据不符合要求。"
        )

    report = {
        "model": args.model,
        "device": "cpu",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "embedding_dimension": len(probe_vector),
        "document_count": len(documents),
        "chunk_count": built.manifest["chunk_count"],
        "persist_dir": str(index_dir),
        "query": query,
        "checks": {
            "reload_completed": True,
            "top_k_count": len(results),
            "scores_descending": scores_descending,
            "top_result_expected": top_result_expected,
            "metadata_complete": metadata_complete,
        },
        "results": [
            {
                "rank": rank,
                "score": result.score,
                "source": result.source_label,
                "metadata": result.metadata,
                "text": result.text,
            }
            for rank, result in enumerate(results, start=1)
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
