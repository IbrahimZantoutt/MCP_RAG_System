"""Cross-encoder reranking — stage 2 of retrieval.

Vector search scores a query against a document embedding computed in advance
and in isolation. A cross-encoder reads the query and the passage *together*
and scores the pair directly. It is far slower, which is why it only ever sees
the RETRIEVE_K candidates that stage 1 already narrowed down to.

This is where questions like "robots stuck at chargers" get resolved correctly.
Pure vector search surfaces the E-401 charge-reservation-timeout passage, which
is the wrong answer: the real condition produces no fault code at all. The
cross-encoder is much better at noticing that distinction.

Disable with HELIX_RERANK=0 to demo the difference.
"""

from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def get_reranker():
    """Load (and cache) the cross-encoder. First call downloads ~90 MB.

    Scores are raw logits, not probabilities: they are unbounded and frequently
    negative. Only their relative order is meaningful.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates against the query and return the best `top_k`.

    Each candidate is a dict with at least a "text" key. The returned dicts gain
    a "rerank_score" and keep their original "score" from vector search, so the
    two rankings can be compared side by side.
    """
    if not candidates:
        return []

    if not config.RERANK_ENABLED:
        return candidates[:top_k]

    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    # Progress bar pinned off; see the note in embedder.embed_query.
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
