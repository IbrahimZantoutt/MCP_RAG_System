"""ChromaDB persistent vector store.

Embeddings are computed by src.embedder and passed in explicitly rather than
letting Chroma manage an embedding function. That keeps one model choice in one
place (config.EMBEDDING_MODEL) and makes the query path identical whether it is
driven by the CLI or by the MCP server.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import config
from .chunker import Chunk


@lru_cache(maxsize=1)
def _client():
    import chromadb
    from chromadb.config import Settings

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(create: bool = True):
    """Return the corpus collection, creating it if needed.

    embedding_function=None is deliberate. src.embedder owns the model, and
    every call site passes vectors in explicitly. Left at its default, Chroma
    would download and instantiate its own ONNX MiniLM (~87 MB) that we would
    never use.
    """
    client = _client()
    if create:
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    return client.get_collection(
        name=config.COLLECTION_NAME, embedding_function=None
    )


def reset_collection() -> None:
    """Drop and recreate the collection. Used by a full reindex."""
    client = _client()
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist
    client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Insert chunks with their precomputed embeddings."""
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
        )

    collection = get_collection()

    # Chroma has a per-call batch ceiling; stay well under it.
    batch = 500
    for i in range(0, len(chunks), batch):
        window = chunks[i:i + batch]
        collection.add(
            ids=[c.chunk_id for c in window],
            documents=[c.text for c in window],
            embeddings=embeddings[i:i + batch],
            metadatas=[c.metadata() for c in window],
        )


def query(
    embedding: list[float],
    n_results: int,
    department: str | None = None,
    source_file: str | None = None,
) -> list[dict[str, Any]]:
    """Vector search, optionally filtered by metadata.

    Returns dicts with text, metadata, and a similarity `score` in [0, 1] where
    higher is better. Chroma reports cosine *distance*, so the conversion is
    1 - distance.
    """
    collection = get_collection()

    where: dict[str, Any] | None = None
    clauses = []
    if department:
        clauses.append({"department": department})
    if source_file:
        clauses.append({"source_file": source_file})
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, max(1, count())),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "chunk_id": cid,
                "text": text,
                "score": 1.0 - float(dist),
                "source_file": meta.get("source_file", ""),
                "department": meta.get("department", ""),
                "doc_title": meta.get("doc_title", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
            }
        )
    return hits


def count() -> int:
    """Number of indexed chunks. Returns 0 if the collection does not exist."""
    try:
        return get_collection(create=False).count()
    except Exception:
        return 0


def list_documents() -> list[dict[str, Any]]:
    """Every indexed document with its department, title, and chunk count.

    Backs the list_documents MCP tool: it gives the model a map of the corpus so
    it can pick a sensible department filter or a document to fetch in full,
    instead of guessing at what exists.
    """
    try:
        collection = get_collection(create=False)
    except Exception:
        return []

    result = collection.get(include=["metadatas"])
    docs: dict[str, dict[str, Any]] = {}

    for meta in result.get("metadatas", []) or []:
        source = meta.get("source_file", "")
        if source not in docs:
            docs[source] = {
                "source_file": source,
                "department": meta.get("department", ""),
                "doc_title": meta.get("doc_title", ""),
                "chunks": 0,
                "tokens": 0,
            }
        docs[source]["chunks"] += 1
        docs[source]["tokens"] += int(meta.get("token_count", 0) or 0)

    return sorted(docs.values(), key=lambda d: (d["department"], d["source_file"]))


def stats() -> dict[str, Any]:
    """Corpus-level index statistics."""
    documents = list_documents()
    by_department: dict[str, dict[str, int]] = {}

    for doc in documents:
        dept = doc["department"]
        entry = by_department.setdefault(dept, {"documents": 0, "chunks": 0})
        entry["documents"] += 1
        entry["chunks"] += doc["chunks"]

    return {
        "collection": config.COLLECTION_NAME,
        "chunks": count(),
        "documents": len(documents),
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_tokens": config.CHUNK_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "by_department": dict(sorted(by_department.items())),
        "storage_path": str(config.CHROMA_DIR),
    }
