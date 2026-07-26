"""The RAG engine.

This is the single implementation of retrieval and grounded answering. Both
front doors use it:

    scripts/ask.py        interactive CLI
    mcp_server/server.py  the search_database tool

They are two interfaces onto one engine, not two copies of the same idea.

Pipeline:

    query
      -> dense search   (bge-small embeddings, cosine)     \
                                                            >- RRF -> candidates
      -> lexical search (BM25 over the same chunks)        /
      -> cross-encoder rerank
      -> RRF again, fusing retrieval rank with rerank rank
      -> assemble numbered context
      -> LLM answers, citing sources by number

Both fusion steps exist for a reason. The first covers dense retrieval's
weakness at exact identifiers. The second stops any single stage from vetoing a
correct result -- an earlier reranker-takes-all design dropped the INC-2024-017
postmortem from vector rank 2 to rank 7, out of the results entirely.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import config, embedder, fusion, lexical, llm, reranker, vector_store

SYSTEM_PROMPT = """\
You are a knowledge assistant for Helix Robotics, an autonomous warehouse robot \
company. You answer questions using only the internal documents provided to you.

Rules:
- Answer only from the numbered context passages. Do not use outside knowledge \
about robotics, other companies, or industry practice.
- Cite the passages you used with their bracketed numbers, like [1] or [2][4]. \
Cite the specific claim, not the whole answer at the end.
- When passages from different documents combine to answer the question, say so \
explicitly and cite each one.
- If the context does not contain the answer, say exactly what is missing. Do \
not guess and do not fill gaps with plausible-sounding detail.
- If passages disagree, surface the disagreement rather than silently picking one.
- Be specific. Prefer the actual figures, version numbers, dates, and names in \
the passages over general summary.
- Keep answers tight. Two or three short paragraphs unless the question needs more.
"""


def retrieve(
    query: str,
    top_k: int | None = None,
    department: str | None = None,
    retrieve_k: int | None = None,
    use_rerank: bool | None = None,
    use_hybrid: bool | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a query.

    Stage 1 builds a candidate pool from dense and lexical search, fused by RRF.
    Stage 2 reranks with a cross-encoder and fuses that ranking back against the
    retrieval ranking.

    Set use_hybrid=False (HELIX_HYBRID=0) for dense-only retrieval, or
    use_rerank=False (HELIX_RERANK=0) to skip stage 2. Both are useful for
    demonstrating what each stage contributes.
    """
    top_k = top_k or config.TOP_K
    retrieve_k = retrieve_k or config.RETRIEVE_K
    if use_rerank is None:
        use_rerank = config.RERANK_ENABLED
    if use_hybrid is None:
        use_hybrid = config.HYBRID_ENABLED

    if vector_store.count() == 0:
        raise RuntimeError(
            "The index is empty. Run: python scripts/index_data.py"
        )

    pool_size = max(retrieve_k, top_k)

    # --- Stage 1: dense + lexical, fused ---------------------------------
    dense_hits = vector_store.query(
        embedding=embedder.embed_query(query),
        n_results=pool_size,
        department=department,
    )

    if use_hybrid:
        lexical_hits = lexical.search(query, pool_size, department=department)
        candidates = fusion.reciprocal_rank_fusion(
            [dense_hits, lexical_hits],
            weights=[config.DENSE_WEIGHT, config.LEXICAL_WEIGHT],
        )[:pool_size]
    else:
        candidates = dense_hits[:pool_size]

    if not candidates:
        return []

    if not use_rerank:
        return candidates[:top_k]

    # --- Stage 2: rerank, then fuse with the retrieval ranking -----------
    reranked = reranker.rerank(query, list(candidates), top_k=len(candidates))
    return fusion.reciprocal_rank_fusion(
        [candidates, reranked],
        weights=[config.RETRIEVAL_WEIGHT, config.RERANK_WEIGHT],
    )[:top_k]


def build_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as numbered, attributed context blocks."""
    blocks = []
    for i, r in enumerate(results, start=1):
        header = (
            f"[{i}] {r['doc_title']} "
            f"({r['department']}/{Path(r['source_file']).name}, "
            f"part {int(r['chunk_index']) + 1} of {r['total_chunks']})"
        )
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return (
        f"Context passages from the Helix Robotics internal corpus:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the passages above, citing them by number."
    )


def _source_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten results into citation records.

    A chunk found only by BM25 has no vector score, and one found only by dense
    search has no BM25 score, so every stage score is optional.
    """
    sources = []
    for i, r in enumerate(results, start=1):
        record = {
            "n": i,
            "source_file": r["source_file"],
            "doc_title": r["doc_title"],
            "department": r["department"],
            "chunk_index": int(r["chunk_index"]),
            "total_chunks": int(r["total_chunks"]),
        }
        for src, dst in (
            ("score", "vector_score"),
            ("bm25_score", "bm25_score"),
            ("rerank_score", "rerank_score"),
            ("rrf_score", "rrf_score"),
        ):
            if r.get(src) is not None:
                record[dst] = round(float(r[src]), 4)
        sources.append(record)
    return sources


def answer(
    question: str,
    top_k: int | None = None,
    department: str | None = None,
    model_key: str | None = None,
    use_rerank: bool | None = None,
    use_hybrid: bool | None = None,
) -> dict[str, Any]:
    """Full RAG: retrieve, then generate a grounded answer with citations."""
    started = time.perf_counter()

    results = retrieve(
        question,
        top_k=top_k,
        department=department,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
    )

    if not results:
        return {
            "question": question,
            "answer": "Nothing in the corpus matched that question.",
            "sources": [],
            "model": llm.resolve_model(model_key),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }

    context = build_context(results)
    retrieval_s = time.perf_counter() - started

    text = llm.generate(
        prompt=build_prompt(question, context),
        system=SYSTEM_PROMPT,
        model_key=model_key,
    )

    return {
        "question": question,
        "answer": text,
        "sources": _source_records(results),
        "model": llm.resolve_model(model_key),
        "retrieval_s": round(retrieval_s, 2),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def answer_stream(
    question: str,
    top_k: int | None = None,
    department: str | None = None,
    model_key: str | None = None,
    use_rerank: bool | None = None,
    use_hybrid: bool | None = None,
):
    """Streaming variant. Yields ("sources", list) first, then ("token", str)."""
    results = retrieve(
        question,
        top_k=top_k,
        department=department,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
    )
    yield ("sources", _source_records(results))

    if not results:
        yield ("token", "Nothing in the corpus matched that question.")
        return

    for piece in llm.stream(
        prompt=build_prompt(question, build_context(results)),
        system=SYSTEM_PROMPT,
        model_key=model_key,
    ):
        yield ("token", piece)


def fetch_document(source_file: str) -> dict[str, Any] | None:
    """Return a full document by its corpus-relative path.

    Retrieval hands the model a ~400-token window. When that window is clearly
    the right document but cut off, this is how it reads the whole thing.
    Backs the fetch_document MCP tool.
    """
    candidate = (config.DATA_DIR / source_file).resolve()
    data_root = config.DATA_DIR.resolve()

    # Never serve anything outside data/, regardless of what was passed in.
    if not candidate.is_relative_to(data_root):
        return None
    if not candidate.is_file() or candidate.suffix != ".txt":
        return None

    text = candidate.read_text(encoding="utf-8")
    rel = candidate.relative_to(data_root)

    return {
        "source_file": rel.as_posix(),
        "department": rel.parts[0] if len(rel.parts) > 1 else "general",
        "text": text,
        "characters": len(text),
        "words": len(text.split()),
    }
