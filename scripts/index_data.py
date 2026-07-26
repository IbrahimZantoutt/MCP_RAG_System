"""Indexing phase — build the vector database from data/.

Loads every .txt under data/, chunks it, embeds the chunks with
sentence-transformers, and writes them to the local Chroma store.

    python scripts/index_data.py            # full rebuild
    python scripts/index_data.py --stats    # inspect the existing index
    python scripts/index_data.py --dry-run  # chunk only, no embedding

Safe to re-run: a rebuild drops and recreates the collection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chunker, config, embedder, vector_store  # noqa: E402


def print_stats() -> None:
    s = vector_store.stats()
    if not s["chunks"]:
        print("Index is empty. Run: python scripts/index_data.py")
        return

    print(f"\nCollection    : {s['collection']}")
    print(f"Documents     : {s['documents']}")
    print(f"Chunks        : {s['chunks']}")
    print(f"Embed model   : {s['embedding_model']}")
    print(f"Chunk budget  : {s['chunk_tokens']} tokens "
          f"({s['chunk_overlap_tokens']} overlap)")
    print(f"Storage       : {s['storage_path']}")
    print("\nBy department:")
    for dept, entry in s["by_department"].items():
        print(f"  {dept:<14} {entry['documents']:>3} docs  {entry['chunks']:>5} chunks")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the Helix corpus.")
    parser.add_argument("--stats", action="store_true",
                        help="show existing index statistics and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk documents but do not embed or write")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return 0

    started = time.perf_counter()

    # --- 1. Load ----------------------------------------------------------
    print(f"Loading documents from {config.DATA_DIR} ...")
    docs = chunker.load_documents()
    if not docs:
        print(f"No .txt files found under {config.DATA_DIR}", file=sys.stderr)
        return 1
    print(f"  {len(docs)} documents")

    # --- 2. Chunk ---------------------------------------------------------
    print(f"\nChunking (budget {config.CHUNK_TOKENS} tokens, "
          f"overlap {config.CHUNK_OVERLAP_TOKENS}) ...")
    chunks = chunker.chunk_all(docs)

    by_dept: dict[str, int] = {}
    for doc in docs:
        by_dept[doc.department] = by_dept.get(doc.department, 0) + len(doc.chunks)

    for dept in sorted(by_dept):
        docs_in_dept = sum(1 for d in docs if d.department == dept)
        print(f"  {dept:<14} {docs_in_dept:>3} docs  {by_dept[dept]:>5} chunks")

    token_counts = [c.token_count for c in chunks]
    print(f"\n  {len(chunks)} chunks total")
    print(f"  tokens/chunk: min {min(token_counts)}, "
          f"mean {sum(token_counts) // len(token_counts)}, "
          f"max {max(token_counts)}")

    oversized = [c for c in chunks if c.token_count > config.CHUNK_TOKENS]
    if oversized:
        print(f"  WARNING: {len(oversized)} chunks exceed the token budget",
              file=sys.stderr)

    if args.dry_run:
        print("\nDry run — nothing embedded or written.")
        return 0

    # --- 3. Embed ---------------------------------------------------------
    print(f"\nEmbedding with {config.EMBEDDING_MODEL} ...")
    texts = [c.embedding_text() for c in chunks]
    embed_started = time.perf_counter()
    embeddings = embedder.embed_documents(texts, show_progress=True)
    embed_elapsed = time.perf_counter() - embed_started
    print(f"  {len(embeddings)} vectors x {len(embeddings[0])} dims "
          f"in {embed_elapsed:.1f}s "
          f"({len(embeddings) / embed_elapsed:.0f} chunks/s)")

    # --- 4. Store ---------------------------------------------------------
    print(f"\nWriting to Chroma at {config.CHROMA_DIR} ...")
    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    print(f"  {vector_store.count()} chunks indexed")

    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    print("\nTry it:  python scripts/ask.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
