# Architecture and interview notes

## Design goals

The project keeps parsing, cleaning, chunking, indexing, retrieval, and LLM generation as separate layers. This makes each stage testable without downloading a model or calling a paid API.

## Data flow

1. `MultiFormatDocumentLoader` converts every source unit into a LlamaIndex `Document`.
2. `clean_documents` normalizes text without discarding paragraph boundaries.
3. `split_documents` calls LlamaIndex `SentenceSplitter` and copies source metadata to every `TextNode`.
4. `PersistentRAGIndex.build` creates a LlamaIndex `VectorStoreIndex`, persists its storage context, and writes a small manifest.
5. `PersistentRAGIndex.load` reloads the index with the same Embedding model.
6. `RAGService` performs Top-K retrieval, applies a minimum cosine similarity, builds a grounded prompt, and returns both the answer and source chunks.

The service treats both retrieved text and the user's question as untrusted
input. Angle brackets are converted to full-width characters before prompt
assembly so a document cannot forge the structural delimiter tags. This is a
defense-in-depth measure, not a complete solution to indirect prompt injection.

## Metadata contract

Every parsed source unit has a deterministic `source_id` derived from its relative source path and locator. Every chunk keeps that ID and adds `chunk_index`.

```text
common: source_file, file_type, source_id, chunk_index
PDF:    page_number
PPTX:   slide_number
CSV:    row_number
```

Absolute local paths are deliberately excluded from the index.

## Similarity semantics

LlamaIndex's local vector store returns cosine similarity, where a larger score indicates a closer match. `min_similarity` therefore retains `score >= threshold`. It is never transformed with `1 - score` and is not handled like an L2 distance.

## Test isolation

Tests use LlamaIndex `MockEmbedding` and a local stub LLM. Synthetic documents are generated inside pytest temporary directories, so tests require neither network access nor credentials.

The separate `scripts/validate_real_embedding.py` smoke test is intentionally
excluded from CI because it downloads the configured model. It creates only
synthetic Chinese text below an ignored `.validation/` directory.
