"""Document loading and paragraph-aware chunking.

Splitting strategy, in priority order:

  1. Never split mid-sentence.
  2. Prefer to split on blank lines (paragraph boundaries). The corpus is full
     of indented spec blocks and procedure steps that lose their meaning when
     cut in half.
  3. Only fall back to sentence-level splitting when a single paragraph is
     itself larger than the token budget.

Token counts come from the embedding model's own tokenizer, so the budget in
config is the real budget rather than a word-count guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config


@dataclass
class Chunk:
    """One indexed unit of text, plus everything needed to cite it."""

    chunk_id: str
    text: str
    source_file: str      # repo-relative, e.g. "engineering/postmortem_INC-2024-017.txt"
    department: str
    doc_title: str
    chunk_index: int
    total_chunks: int = 0
    token_count: int = 0

    def embedding_text(self) -> str:
        """Text actually handed to the embedding model.

        A short contextual header is prepended so that a chunk from the middle
        of a document still carries the signal of what document it came from.
        A chunk reading only "See section 4" is useless in isolation; the same
        chunk headed "Fleet Recovery Runbook (engineering)" is retrievable.
        """
        return f"{self.doc_title} [{self.department}]\n\n{self.text}"

    def metadata(self) -> dict:
        """Chroma metadata payload. Chroma accepts only scalar values."""
        return {
            "source_file": self.source_file,
            "department": self.department,
            "doc_title": self.doc_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
        }


@dataclass
class Document:
    path: Path
    source_file: str
    department: str
    title: str
    text: str
    chunks: list[Chunk] = field(default_factory=list)


@lru_cache(maxsize=1)
def _tokenizer():
    """The embedding model's tokenizer, loaded once."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


_BANNER = re.compile(r"^HELIX ROBOTICS(?:,\s*INC\.)?\s*[—–-]\s*(.+)$", re.IGNORECASE)

_ACRONYMS = {
    "SLA", "FAQ", "HR", "IT", "CEO", "CTO", "CFO", "EMEA", "ISO", "OS",
    "WMS", "MCP", "AMR", "RF", "SSO", "PTO", "QA", "SRE", "TAM", "DC",
}
_SMALL_WORDS = {"and", "or", "of", "the", "for", "to", "a", "an", "in", "on", "at"}


def _smart_title_case(banner: str) -> str:
    """Turn a shouted banner into a readable title.

    "SUPPORT SLA AND ESCALATION POLICY" -> "Support SLA and Escalation Policy"

    Cosmetic only: the embedding model is uncased, so this affects citations
    rather than retrieval.
    """
    words = banner.split()
    out: list[str] = []

    for i, word in enumerate(words):
        stripped = word.strip(".,")

        if any(ch.isdigit() for ch in word):        # 2025, HX-200, Q3
            out.append(word)
        elif stripped.upper() in _ACRONYMS:
            out.append(word.upper())
        elif i > 0 and stripped.lower() in _SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append("-".join(p.capitalize() for p in word.split("-")))

    return " ".join(out)


def _derive_title(text: str, path: Path) -> str:
    """Build a human title from the document header block.

    Corpus documents open with a "HELIX ROBOTICS — <BANNER>" line, usually
    followed by a qualifier that says which document of that kind this is.
    Combining the two gives titles that stay distinguishable once they are
    prepended to every chunk as retrieval context:

        "HX-200 'Courier' Autonomous Mobile Robot - Product Specification"
        "Incident Postmortem INC-2024-017 - Charge dock reservation deadlock..."
        "Support Ticket Log - Q3 2024 (July 1 - September 30)"
    """
    lines = [ln.strip() for ln in text.splitlines()[:14] if ln.strip()]
    if not lines:
        return path.stem.replace("_", " ").title()

    banner = ""
    match = _BANNER.match(lines[0])
    if match:
        banner = _smart_title_case(match.group(1).strip())

    def field(key: str) -> str:
        prefix = key + ":"
        for line in lines[1:]:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return ""

    # A "Document:"/"Title:" value is already a full descriptive title.
    document = field("Document") or field("Title")
    incident = field("Incident")
    if document:
        if incident:
            return f"{banner or 'Incident Postmortem'} {incident} — {document}"
        return document

    # Otherwise qualify the banner with whatever identifies this instance.
    for key in ("Product", "Subject", "Account", "Product line"):
        value = field(key)
        if value:
            return f"{value} — {banner}" if banner else value

    period = field("Period")
    if period and banner:
        return f"{banner} — {period}"

    if incident and banner:
        return f"{banner} {incident}"

    return banner or path.stem.replace("_", " ").title()


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """Load every .txt under data/, treating each subdirectory as a department."""
    data_dir = data_dir or config.DATA_DIR
    docs: list[Document] = []

    for path in sorted(data_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        rel = path.relative_to(data_dir)
        department = rel.parts[0] if len(rel.parts) > 1 else "general"

        docs.append(
            Document(
                path=path,
                source_file=rel.as_posix(),
                department=department,
                title=_derive_title(text, path),
                text=text,
            )
        )

    return docs


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_END.split(paragraph)
    return [p for p in (s.strip() for s in parts) if p]


def _split_oversized(paragraph: str, budget: int) -> list[str]:
    """Break a paragraph that exceeds the budget on sentence boundaries.

    Last resort: a single sentence larger than the budget is split on
    whitespace, which only happens with long tabular blocks.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in _split_sentences(paragraph):
        n = count_tokens(sentence)

        if n > budget:
            if current:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            words = sentence.split()
            step = max(1, len(words) * budget // max(n, 1))
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i:i + step]))
            continue

        if current_tokens + n > budget and current:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0

        current.append(sentence)
        current_tokens += n

    if current:
        pieces.append(" ".join(current))

    return pieces


def _overlap_tail(paragraphs: list[str], overlap_tokens: int) -> list[str]:
    """Take whole trailing paragraphs worth roughly `overlap_tokens`.

    Overlapping on paragraph boundaries rather than a raw token slice keeps the
    carried-over text readable, which matters because it appears verbatim in
    cited results.
    """
    if overlap_tokens <= 0:
        return []

    tail: list[str] = []
    total = 0
    for para in reversed(paragraphs):
        n = count_tokens(para)
        if total + n > overlap_tokens and tail:
            break
        tail.insert(0, para)
        total += n
        if total >= overlap_tokens:
            break
    return tail


def chunk_document(doc: Document) -> list[Chunk]:
    """Split one document into overlapping, paragraph-aligned chunks."""
    budget = config.CHUNK_TOKENS
    overlap = config.CHUNK_OVERLAP_TOKENS

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]

    # Pre-split anything already too large to ever fit.
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        if count_tokens(para) > budget:
            paragraphs.extend(_split_oversized(para, budget))
        else:
            paragraphs.append(para)

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(current)
        n = count_tokens(text)
        if n >= config.MIN_CHUNK_TOKENS or not chunks:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_file}::{idx}",
                    text=text,
                    source_file=doc.source_file,
                    department=doc.department,
                    doc_title=doc.title,
                    chunk_index=idx,
                    token_count=n,
                )
            )
        carry = _overlap_tail(current, overlap)
        current = list(carry)
        current_tokens = sum(count_tokens(p) for p in carry)

    for para in paragraphs:
        n = count_tokens(para)
        if current_tokens + n > budget and current:
            flush()
        current.append(para)
        current_tokens += n

    # Final flush without carrying overlap forward.
    if current:
        text = "\n\n".join(current)
        n = count_tokens(text)
        if n >= config.MIN_CHUNK_TOKENS or not chunks:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_file}::{idx}",
                    text=text,
                    source_file=doc.source_file,
                    department=doc.department,
                    doc_title=doc.title,
                    chunk_index=idx,
                    token_count=n,
                )
            )

    for c in chunks:
        c.total_chunks = len(chunks)

    doc.chunks = chunks
    return chunks


def chunk_all(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc))
    return out
