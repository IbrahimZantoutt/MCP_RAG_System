"""BM25 lexical search — the keyword half of hybrid retrieval.

Dense embeddings are weak at exact identifiers, and this corpus is saturated
with them: INC-2024-017, E-401, TKT-8841, firmware 3.8.1, Fleet OS 4.2.2. All
of those are near-invisible to a 384-dimension sentence embedding, and all of
them are exactly what a user types when they want a specific document.

BM25 is the classic answer and it is small enough to implement directly rather
than take a dependency for.

The index is built lazily from the chunks already stored in Chroma, so there is
no second artifact on disk to keep in sync with the vector store.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from . import config, vector_store

K1 = 1.5   # term-frequency saturation
B = 0.75   # length normalisation

# Keep dots and hyphens *inside* tokens so identifiers survive tokenisation:
# "inc-2024-017", "e-401", "3.8.1", "hx-200" stay whole.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]*")

# Removed because they carry no retrieval signal but do carry BM25 weight. On
# this corpus "what" scored idf 2.046 -- higher than "inc-2024-017" at 1.769 --
# so an unfiltered query was partly ranking on its own question words.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am
do does did doing done
have has had having
i you he she it we they them his her its our your their
of in on at to for from by with without about into over under
as no not so such only own same too very can will just should now
what which who whom when where why how
all any both each few more most other some
me him us my mine yours ours theirs
""".split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase and extract tokens, keeping identifiers intact.

    Compound identifiers are deliberately NOT split into parts. Splitting
    "inc-2024-017" into ["inc", "2024", "017"] looks helpful but injects "2024",
    which appears in 375 of 436 chunks here and dilutes the very term it was
    meant to reinforce.
    """
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        token = match.strip("._-")
        if not token:
            continue
        if drop_stopwords and token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


class BM25Index:
    """Okapi BM25 over the indexed chunks."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.doc_tokens = [tokenize(r["_index_text"]) for r in records]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.n = len(records)

        # term -> {doc index: term frequency}
        self.postings: dict[str, dict[int, int]] = {}
        for i, tokens in enumerate(self.doc_tokens):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, {})[i] = freq

        self.idf: dict[str, float] = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, n_results: int) -> list[dict[str, Any]]:
        scores: dict[int, float] = {}

        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_id, freq in posting.items():
                norm = 1 - B + B * (self.doc_len[doc_id] / (self.avg_len or 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    freq * (K1 + 1) / (freq + K1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_results]

        hits = []
        for doc_id, score in ranked:
            record = dict(self.records[doc_id])
            record.pop("_index_text", None)
            record["bm25_score"] = round(score, 4)
            hits.append(record)
        return hits


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    """Build (and cache) the BM25 index from the Chroma collection.

    Indexes title + department + body, matching what the embedder sees, so a
    query naming a document by title hits lexically too.
    """
    try:
        collection = vector_store.get_collection(create=False)
        result = collection.get(include=["documents", "metadatas"])
    except Exception:
        return BM25Index([])

    records: list[dict[str, Any]] = []
    ids = result.get("ids", []) or []
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []

    for cid, text, meta in zip(ids, documents, metadatas):
        title = meta.get("doc_title", "")
        department = meta.get("department", "")
        records.append(
            {
                "chunk_id": cid,
                "text": text,
                "source_file": meta.get("source_file", ""),
                "department": department,
                "doc_title": title,
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
                "_index_text": f"{title} {department} {text}",
            }
        )

    return BM25Index(records)


def search(
    query: str,
    n_results: int,
    department: str | None = None,
) -> list[dict[str, Any]]:
    """BM25 search, optionally restricted to one department."""
    index = get_index()
    if index.n == 0:
        return []

    # Over-fetch before filtering so a department filter cannot starve results.
    hits = index.search(query, n_results * 4 if department else n_results)
    if department:
        hits = [h for h in hits if h["department"] == department]
    return hits[:n_results]


def reset_cache() -> None:
    """Drop the cached index. Call after reindexing inside a live process."""
    get_index.cache_clear()
