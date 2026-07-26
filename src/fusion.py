"""Reciprocal Rank Fusion.

RRF combines several rankings of the same items without needing their scores to
be on comparable scales — which matters here, because cosine similarity (0..1),
BM25 (unbounded positive), and cross-encoder logits (unbounded, often negative)
cannot be averaged directly.

    score(d) = sum over rankers of  1 / (k + rank(d))

The constant k (60 by convention) damps the influence of top ranks so that one
ranker placing an item first cannot dominate the consensus.

Using RRF twice is deliberate. Once to combine dense and lexical retrieval, and
again to combine that consensus with the cross-encoder. The second use is what
stops a single stage from vetoing a correct result: an item ranked #2 by
retrieval and #7 by the reranker still finishes near the top, instead of being
dropped outright the way it was under reranker-takes-all.
"""

from __future__ import annotations

from typing import Any, Iterable

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Iterable[list[dict[str, Any]]],
    key: str = "chunk_id",
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked lists into one, best first.

    Items are matched across lists by `key`. The merged record keeps whichever
    fields each list contributed (vector score, bm25 score, rerank score), plus
    an "rrf_score" and the per-ranker ranks it was built from.
    """
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)

    merged: dict[Any, dict[str, Any]] = {}
    scores: dict[Any, float] = {}
    ranks: dict[Any, dict[int, int]] = {}

    for source, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, item in enumerate(ranking, start=1):
            item_key = item[key]

            if item_key not in merged:
                merged[item_key] = dict(item)
            else:
                # Fill in fields this list has that earlier lists did not.
                for field, value in item.items():
                    merged[item_key].setdefault(field, value)

            scores[item_key] = scores.get(item_key, 0.0) + weight / (k + rank)
            ranks.setdefault(item_key, {})[source] = rank

    for item_key, record in merged.items():
        record["rrf_score"] = round(scores[item_key], 6)
        record["ranks"] = ranks[item_key]

    return sorted(merged.values(), key=lambda r: r["rrf_score"], reverse=True)
