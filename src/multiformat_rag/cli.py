"""Command-line interface for indexing, retrieval, and RAG answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multiformat_rag.chunking import split_documents
from multiformat_rag.config import AppConfig
from multiformat_rag.exceptions import RAGError
from multiformat_rag.indexing import PersistentRAGIndex, SearchResult
from multiformat_rag.llm import DashScopeLLMClient
from multiformat_rag.loaders import MultiFormatDocumentLoader
from multiformat_rag.preprocessing import clean_documents
from multiformat_rag.service import RAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multiformat-rag",
        description="Python 多格式文档 RAG：索引、检索与可追溯问答。",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="可选 .env 文件路径；密钥不会输出到终端。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="解析文档并构建本地索引")
    index_parser.add_argument("input", type=Path, help="文档文件或目录")
    index_parser.add_argument("--persist-dir", type=Path, default=None)
    index_parser.add_argument("--embedding-model", default=None)
    index_parser.add_argument("--chunk-size", type=int, default=None)
    index_parser.add_argument("--chunk-overlap", type=int, default=None)
    index_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名索引文件；不会删除索引目录。",
    )

    retrieve_parser = subparsers.add_parser("retrieve", help="执行 Top-K 语义检索")
    _add_query_arguments(retrieve_parser)
    retrieve_parser.add_argument("--json", action="store_true", help="输出 JSON")

    ask_parser = subparsers.add_parser("ask", help="检索文档并调用 LLM 回答")
    _add_query_arguments(ask_parser)
    ask_parser.add_argument("--llm-model", default=None)
    ask_parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", help="检索或提问内容")
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-similarity", type=float, default=None)


def _create_embedding(model_name: str):
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise RAGError(
            "未安装 HuggingFace Embedding 集成，请按 README 安装项目依赖。"
        ) from exc
    print(f"正在加载 Embedding 模型：{model_name}", file=sys.stderr)
    try:
        return HuggingFaceEmbedding(model_name=model_name)
    except Exception as exc:
        raise RAGError(
            "Embedding 模型加载失败。请检查模型名称、网络连接、本地缓存和可用内存。"
        ) from exc


def _load_index(args, config: AppConfig) -> PersistentRAGIndex:
    persist_dir = args.persist_dir or config.index_dir
    manifest = PersistentRAGIndex.read_manifest(persist_dir)
    model_name = args.embedding_model or str(manifest["embedding_model"])
    embed_model = _create_embedding(model_name)
    return PersistentRAGIndex.load(
        persist_dir,
        embed_model=embed_model,
        expected_embedding_model=model_name,
    )


def _result_payload(result: SearchResult) -> dict:
    return {
        "text": result.text,
        "score": result.score,
        "source": result.source_label,
        "metadata": result.metadata,
    }


def _run_index(args, config: AppConfig) -> int:
    model_name = args.embedding_model or config.embedding_model
    chunk_size = config.chunk_size if args.chunk_size is None else args.chunk_size
    chunk_overlap = (
        config.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap
    )
    persist_dir = args.persist_dir or config.index_dir

    raw_documents = MultiFormatDocumentLoader().load(args.input)
    documents = clean_documents(raw_documents)
    nodes = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embed_model = _create_embedding(model_name)
    index = PersistentRAGIndex.build(
        nodes,
        persist_dir=persist_dir,
        embed_model=embed_model,
        embedding_model_name=model_name,
        allow_overwrite=args.overwrite,
        show_progress=True,
    )
    print(
        f"索引构建完成：{index.manifest['chunk_count']} 个分块，"
        f"{len(index.manifest['source_files'])} 个文件，目录 {index.persist_dir}"
    )
    return 0


def _run_retrieve(args, config: AppConfig) -> int:
    index = _load_index(args, config)
    service = RAGService(index)
    top_k = config.top_k if args.top_k is None else args.top_k
    min_similarity = (
        config.min_similarity
        if args.min_similarity is None
        else args.min_similarity
    )
    results = service.retrieve(
        args.question,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    if args.json:
        print(json.dumps([_result_payload(item) for item in results], ensure_ascii=False, indent=2))
    elif not results:
        print("没有达到相似度阈值的检索结果。")
    else:
        for rank, result in enumerate(results, start=1):
            score = "N/A" if result.score is None else f"{result.score:.4f}"
            print(f"[{rank}] score={score} | {result.source_label}")
            print(result.text)
            print()
    return 0


def _run_ask(args, config: AppConfig) -> int:
    api_key = config.require_dashscope_api_key()
    index = _load_index(args, config)
    service = RAGService(index)
    llm = DashScopeLLMClient(
        model_name=args.llm_model or config.llm_model,
        api_key=api_key,
    )
    result = service.answer(
        args.question,
        llm=llm,
        top_k=config.top_k if args.top_k is None else args.top_k,
        min_similarity=(
            config.min_similarity
            if args.min_similarity is None
            else args.min_similarity
        ),
    )
    if args.json:
        payload = {
            "answer": result.answer,
            "sources": [_result_payload(item) for item in result.sources],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        if result.sources:
            print("\n来源：")
            for number, source in enumerate(result.sources, start=1):
                score = "N/A" if source.score is None else f"{source.score:.4f}"
                print(f"[{number}] {source.source_label}（score={score}）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = AppConfig.from_env(args.env_file)
        if args.command == "index":
            return _run_index(args, config)
        if args.command == "retrieve":
            return _run_retrieve(args, config)
        return _run_ask(args, config)
    except (RAGError, ValueError, FileExistsError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
