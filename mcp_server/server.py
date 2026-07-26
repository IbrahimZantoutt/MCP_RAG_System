"""Helix Robotics MCP server.

Exposes the RAG system as Model Context Protocol tools over stdio.

    python mcp_server/server.py

Four tools:

    search_database   semantic + lexical search over the internal corpus
    fetch_document    read a full document once search has identified it
    list_documents    map of what exists, so the model stops guessing
    search_web        live external search, for anything not internal

A note on search_database: it returns retrieved passages rather than a
finished answer. The MCP client is itself a language model, so generating an
answer inside the tool would mean running a second, weaker model whose summary
discards detail the calling model needs. Retrieval is the tool's job; synthesis
is the caller's. scripts/ask.py is the path that does both, for using the
system standalone.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Protect stdout ------------------------------------------------------
# Under stdio transport, stdout IS the JSON-RPC channel. A single stray byte
# from a library -- a tqdm progress bar, a stray print, a warning -- corrupts
# the stream, the client cannot parse the reply, and the session hangs forever
# rather than failing. This must run before anything heavy is imported.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Keep the real stdout for the protocol, and point every other writer at
# stderr, where a client is happy to see diagnostics.
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr

for _noisy in ("sentence_transformers", "transformers", "chromadb", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src import config, lexical, rag, vector_store  # noqa: E402

mcp = FastMCP("helix-rag")

MAX_WEB_RESULTS = 10
MAX_DOCUMENT_CHARS = 60_000


@mcp.tool()
def search_database(
    query: str,
    department: str | None = None,
    top_k: int = 5,
) -> str:
    """Search Helix Robotics' internal knowledge base.

    Covers engineering documentation, incident postmortems, operational
    runbooks, product specifications, support tickets and troubleshooting
    guides, sales account records, pricing, HR policy, financial reports, and
    safety and compliance material.

    Uses hybrid retrieval: dense vector search for meaning, BM25 for exact
    identifiers such as INC-2024-017, E-401, or firmware 3.8.1, fused and then
    reranked by a cross-encoder.

    Answers frequently require combining passages from several documents. When
    the results are partial, search again with different wording, or call
    fetch_document to read a promising source in full.

    Args:
        query: A natural-language question or keywords. Specific is better;
            include identifiers, version numbers, and names where you have them.
        department: Optionally restrict to one of: compliance, engineering,
            finance, hr, product, sales, support. Omit to search everything.
        top_k: Number of passages to return (1-20, default 5).
    """
    query = (query or "").strip()
    if not query:
        return "Error: query must not be empty."

    if department and department not in config.DEPARTMENTS:
        return (
            f"Error: unknown department {department!r}. "
            f"Valid values: {', '.join(config.DEPARTMENTS)}"
        )

    top_k = max(1, min(int(top_k), 20))

    try:
        results = rag.retrieve(query, top_k=top_k, department=department)
    except RuntimeError as exc:
        return f"Error: {exc}"

    if not results:
        scope = f" in department '{department}'" if department else ""
        return (
            f"No passages matched{scope}. Try broader wording, drop the "
            f"department filter, or call list_documents to see what exists."
        )

    lines = [
        f"{len(results)} passage(s) for: {query}",
        "",
    ]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] {r['doc_title']}\n"
            f"    source: {r['source_file']}  "
            f"(part {int(r['chunk_index']) + 1} of {r['total_chunks']})\n"
        )
        lines.append(r["text"])
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    lines.append(
        "Cite sources by their file path. To read any of these documents in "
        "full, call fetch_document with its source path."
    )
    return "\n".join(lines)


@mcp.tool()
def fetch_document(source_file: str) -> str:
    """Read a complete document from the Helix internal corpus.

    search_database returns roughly 400-token excerpts. When an excerpt is
    clearly from the right document but is cut off, or when you need the full
    context around it, fetch the whole document here.

    Args:
        source_file: Corpus-relative path exactly as reported by
            search_database or list_documents, for example
            "engineering/postmortem_INC-2024-017.txt".
    """
    source_file = (source_file or "").strip().replace("\\", "/")
    if not source_file:
        return "Error: source_file must not be empty."

    document = rag.fetch_document(source_file)
    if document is None:
        return (
            f"Error: no document at {source_file!r}. Paths look like "
            f"'engineering/postmortem_INC-2024-017.txt'. "
            f"Call list_documents to see valid paths."
        )

    text = document["text"]
    truncated = ""
    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS]
        truncated = (
            f"\n\n[Truncated at {MAX_DOCUMENT_CHARS} characters of "
            f"{document['characters']}. Use search_database to target the "
            f"remainder.]"
        )

    return (
        f"{document['source_file']}  "
        f"(department: {document['department']}, {document['words']} words)\n"
        f"{'=' * 70}\n\n"
        f"{text}{truncated}"
    )


@mcp.tool()
def list_documents(department: str | None = None) -> str:
    """List every document in the Helix internal corpus.

    Use this to orient yourself before searching: it shows what subject matter
    exists and which department owns it, so you can choose a sensible
    department filter or identify a document to fetch in full.

    Args:
        department: Optionally restrict to one of: compliance, engineering,
            finance, hr, product, sales, support. Omit to list everything.
    """
    if department and department not in config.DEPARTMENTS:
        return (
            f"Error: unknown department {department!r}. "
            f"Valid values: {', '.join(config.DEPARTMENTS)}"
        )

    documents = vector_store.list_documents()
    if not documents:
        return "The index is empty. Run: python scripts/index_data.py"

    if department:
        documents = [d for d in documents if d["department"] == department]
        if not documents:
            return f"No documents in department {department!r}."

    stats = vector_store.stats()
    lines = [
        f"{len(documents)} document(s), {sum(d['chunks'] for d in documents)} "
        f"indexed passages.",
        "",
    ]

    current = None
    for doc in documents:
        if doc["department"] != current:
            current = doc["department"]
            lines.append(f"{current.upper()}")
        lines.append(f"  {doc['source_file']}")
        lines.append(f"      {doc['doc_title']}  ({doc['chunks']} passages)")

    if not department:
        lines.append("")
        lines.append(
            f"Departments: {', '.join(stats['by_department'])}"
        )
    return "\n".join(lines)


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """Search the public web.

    For information that is not in Helix's internal corpus: industry standards,
    competitor announcements, current events, general technical reference.

    For anything about Helix Robotics itself -- its products, customers,
    incidents, policies, or finances -- use search_database instead. Helix
    internal information is not on the public web.

    Args:
        query: Search terms.
        max_results: Number of results to return (1-10, default 5).
    """
    query = (query or "").strip()
    if not query:
        return "Error: query must not be empty."

    max_results = max(1, min(int(max_results), MAX_WEB_RESULTS))

    try:
        from ddgs import DDGS

        with DDGS() as engine:
            results = list(engine.text(query, max_results=max_results))
    except Exception as exc:
        return (
            f"Web search failed: {type(exc).__name__}: {exc}\n"
            f"This tool needs network access. The internal corpus is still "
            f"available through search_database."
        )

    if not results:
        return f"No web results for: {query}"

    lines = [f"{len(results)} web result(s) for: {query}", ""]
    for i, r in enumerate(results, start=1):
        title = r.get("title") or "(untitled)"
        url = r.get("href") or r.get("url") or ""
        body = (r.get("body") or "").strip()
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    {url}")
        if body:
            lines.append(f"    {body}")
        lines.append("")

    return "\n".join(lines)


def warmup() -> None:
    """Load models and indexes before the protocol starts.

    Runs while sys.stdout still points at stderr, so any chatter from model
    loading cannot reach the protocol stream. It also means the first
    search_database call is fast instead of paying a 10-second model load.
    """
    if vector_store.count() == 0:
        print(
            "Warning: the vector index is empty. "
            "Run: python scripts/index_data.py",
            file=sys.stderr,
        )
        return

    lexical.get_index()
    try:
        rag.retrieve("warmup", top_k=1)
    except Exception as exc:  # non-fatal: tools still work, just slower
        print(f"Warmup retrieval failed: {exc}", file=sys.stderr)


def main() -> None:
    warmup()

    # Hand the real stdout back for the JSON-RPC channel. This must happen
    # before mcp.run(), because the stdio transport reads sys.stdout.buffer to
    # find its output stream.
    sys.stdout = _PROTOCOL_STDOUT
    mcp.run()


if __name__ == "__main__":
    main()
