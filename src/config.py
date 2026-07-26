"""Central configuration.

Every tunable in the project lives here. Environment variables override the
defaults, so you can change behaviour without editing code:

    HELIX_MODEL=minimax python scripts/ask.py
    HELIX_RERANK=0 python scripts/ask.py
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"

COLLECTION_NAME = "helix_corpus"


# --- Language models -------------------------------------------------------
# Both are served through Ollama. Add more by extending this dict; nothing else
# needs to change.

MODELS: dict[str, str] = {
    "deepseek": "deepseek-coder-v2:lite",   # local, default
    "minimax": "minimax-m2.5:cloud",        # cloud, stronger at prose synthesis
}

# Which key from MODELS to use. Override with HELIX_MODEL.
ACTIVE_MODEL = os.getenv("HELIX_MODEL", "deepseek")

# Generation parameters. Low temperature: this is a grounded-answer task, not a
# creative one, and we want the model to stick to the retrieved context.
LLM_TEMPERATURE = float(os.getenv("HELIX_TEMPERATURE", "0.1"))
LLM_NUM_CTX = int(os.getenv("HELIX_NUM_CTX", "8192"))
LLM_TIMEOUT_S = int(os.getenv("HELIX_TIMEOUT", "300"))


# --- Embeddings ------------------------------------------------------------
# bge-small-en-v1.5: 384 dimensions, ~130 MB, CPU-friendly, and meaningfully
# better at retrieval than all-MiniLM-L6-v2 at the same size.

EMBEDDING_MODEL = os.getenv("HELIX_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64

# bge-*-en-v1.5 models expect short queries to carry this instruction prefix.
# Documents are embedded without it. Skipping this costs real retrieval quality.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# --- Chunking --------------------------------------------------------------
# Budget is in tokens, measured with the embedding model's own tokenizer.
#
# bge-small has a 512-token limit. We chunk to 400 so that the contextual header
# prepended at embedding time (title + department) cannot push a chunk over the
# limit and cause silent truncation.

CHUNK_TOKENS = int(os.getenv("HELIX_CHUNK_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("HELIX_CHUNK_OVERLAP", "60"))
MIN_CHUNK_TOKENS = 40  # discard fragments smaller than this


# --- Retrieval -------------------------------------------------------------
# Over-fetch with the cheap vector search, then let the cross-encoder pick the
# best few. This two-stage shape is the single biggest quality win available.

RETRIEVE_K = int(os.getenv("HELIX_RETRIEVE_K", "30"))   # stage 1: candidate pool
TOP_K = int(os.getenv("HELIX_TOP_K", "5"))              # stage 2: after rerank

# Hybrid retrieval: BM25 lexical search fused with dense vector search.
#
# The corpus is full of exact identifiers -- INC-2024-017, E-401, TKT-8841,
# firmware 3.8.1 -- which dense embeddings represent poorly and BM25 handles
# natively. Cosine scores across this corpus span only ~0.61 to 0.67, so dense
# retrieval alone barely discriminates; the lexical signal is what separates
# "the document about this incident" from "a document that mentions it".
HYBRID_ENABLED = os.getenv("HELIX_HYBRID", "1") not in ("0", "false", "False")
DENSE_WEIGHT = float(os.getenv("HELIX_DENSE_WEIGHT", "1.0"))
LEXICAL_WEIGHT = float(os.getenv("HELIX_LEXICAL_WEIGHT", "1.0"))

# Weight of the cross-encoder's ranking when fused with the retrieval ranking.
# Fusing rather than replacing means the reranker can promote and demote, but
# cannot single-handedly drop a strong retrieval hit out of the results.
RERANK_WEIGHT = float(os.getenv("HELIX_RERANK_WEIGHT", "1.0"))
RETRIEVAL_WEIGHT = float(os.getenv("HELIX_RETRIEVAL_WEIGHT", "1.0"))

RERANK_ENABLED = os.getenv("HELIX_RERANK", "1") not in ("0", "false", "False")

# ms-marco-MiniLM-L-6-v2: ~90 MB, fast on CPU, and a strong reranker for its
# size. Chosen deliberately over BAAI/bge-reranker-base, which scores a little
# better but is a BERT-base cross-encoder at ~1.1 GB and noticeably slower on a
# laptop already hosting a local LLM. Swap it in if you want the extra quality:
#     HELIX_RERANK_MODEL=BAAI/bge-reranker-base
RERANK_MODEL = os.getenv(
    "HELIX_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --- Departments -----------------------------------------------------------
# Derived from the data/ subdirectory names. Used for metadata filtering and
# exposed as an enum on the search_database MCP tool.

DEPARTMENTS = [
    "compliance",
    "engineering",
    "finance",
    "hr",
    "product",
    "sales",
    "support",
]


def active_model_name() -> str:
    """Resolve ACTIVE_MODEL to the Ollama model tag."""
    if ACTIVE_MODEL not in MODELS:
        raise ValueError(
            f"Unknown model key {ACTIVE_MODEL!r}. "
            f"Available: {', '.join(sorted(MODELS))}"
        )
    return MODELS[ACTIVE_MODEL]
