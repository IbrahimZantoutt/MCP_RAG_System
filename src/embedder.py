"""Sentence-transformer embeddings.

The model is loaded lazily and cached, because both the indexing script and the
long-running MCP server import this module and neither should pay the load cost
until it actually embeds something.

Vectors are L2-normalised, so cosine similarity is a dot product and Chroma's
cosine space behaves predictably.
"""

from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def get_model():
    """Load (and cache) the SentenceTransformer. First call downloads ~130 MB."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_documents(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed passages for indexing. No instruction prefix — bge wants none here."""
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a search query.

    bge-*-en-v1.5 is trained with an asymmetric instruction prefix on the query
    side only. Omitting it measurably degrades retrieval, and it is the single
    easiest thing to get wrong with this model family.

    show_progress_bar is pinned off. Left unset, sentence-transformers decides
    based on the effective log level, and under the MCP server -- which
    configures logging at INFO -- it turns itself on and writes a progress bar
    that can land on stdout, the same stream the MCP protocol uses.
    """
    model = get_model()
    vector = model.encode(
        config.QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vector.tolist()
