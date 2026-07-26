# Helix RAG + MCP Server

A local retrieval-augmented generation system over a synthetic corporate corpus,
exposed both as a CLI and as a Model Context Protocol server.

Everything runs locally — embeddings, vector store, reranking, and the LLM. No
API keys required.

---

## Demo

Answering a question whose answer exists in no single document — the system
combines passages from several files and cites each one.

<!-- VIDEO
     Easiest: open a GitHub issue, drag the .mp4 in, copy the generated
     https://github.com/user-attachments/... URL, and paste it on its own line
     below. GitHub renders it as an inline player.
     Alternative: commit the file and use <video src="docs/demo.mp4" controls>.
-->

<img width="1318" height="628" alt="rag_answering" src="https://github.com/user-attachments/assets/2ca224ae-3185-40b4-a2f5-26ca2432495b" />

<table>
<tr>
<td width="50%" valign="top">

**Sources cited in the answer**

<img src="https://github.com/user-attachments/assets/74f9b486-f030-4b5e-ac1d-ee0076fbfb57" alt="Cited passages with vector, BM25, rerank and RRF scores" width="100%">

</td>
<td width="50%" valign="top">

**The same passage in the original document**

<img src="https://github.com/user-attachments/assets/ff019350-bca5-445b-8849-c3d6ced5e6be" alt="The cited section shown in its source document" width="100%">

</td>
</tr>
</table>

---

## Quick start

```bash
pip install -r requirements.txt
ollama pull deepseek-coder-v2:lite

python scripts/index_data.py     # build the index (~40s)
python scripts/ask.py            # ask questions
```

First run downloads ~220 MB of models (embedder + reranker).

---

## How it connects

Two pipelines. **Indexing** runs once and writes to disk:

```
   data/*.txt
       │
       ▼
   chunker.py ──── paragraph-aware split, 400-token budget
       │
       ▼
   embedder.py ─── bge-small-en-v1.5 → 384-dim vectors
       │
       ▼
   vector_store.py ─── ChromaDB → storage/chroma/
```

**Querying** runs per question. Both front doors enter the same engine:

```
   scripts/ask.py          mcp_server/server.py
          │                        │
          └───────────┬────────────┘
                      ▼
                   rag.py                    ← the engine
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   embedder.py                  lexical.py
   + vector_store.py            (BM25)
   (dense search)               (exact terms)
        │                           │
        └─────────────┬─────────────┘
                      ▼
                  fusion.py                  ← RRF
                      │
                      ▼
                 reranker.py                 ← cross-encoder
                      │
                      ▼
                  fusion.py                  ← RRF again
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
     llm.py                    return passages
   (Ollama) → answer           → MCP client answers
```

The two paths share retrieval exactly. They differ only in who writes the
answer: `ask.py` uses a local Ollama model, while the MCP server returns
passages and lets the calling model synthesize. **No LLM runs inside the MCP
server.**

### Modules

| File | Role |
|---|---|
| `src/config.py` | Every tunable, overridable by env var |
| `src/chunker.py` | Document loading, title extraction, chunking |
| `src/embedder.py` | Embedding model wrapper |
| `src/vector_store.py` | ChromaDB persistence and metadata filtering |
| `src/lexical.py` | BM25, implemented directly (no extra dependency) |
| `src/fusion.py` | Reciprocal rank fusion |
| `src/reranker.py` | Cross-encoder reranking |
| `src/llm.py` | Ollama client and model switching |
| `src/rag.py` | The engine — retrieve, prompt, grounded answer |
| `mcp_server/server.py` | FastMCP stdio server, 4 tools |

---

## The corpus

27 documents, ~76k words, for a fictional warehouse-robotics company. Seven
departments: `engineering`, `support`, `product`, `sales`, `hr`, `finance`,
`compliance`.

Documents are deliberately cross-referenced. One incident — a charging-dock
deadlock that stopped 61 robots — appears in seven files, each holding a
different piece: root cause, recovery procedure, customer call, commercial
fallout, cost, and the two policies that changed because of it. No single
document tells the whole story, which is what makes retrieval quality visible.

---

## Retrieval

Hybrid, because the corpus is full of exact identifiers (`INC-2024-017`,
`E-401`, `firmware 3.8.1`) that dense embeddings represent poorly — cosine
scores span only 0.61–0.67 across the whole corpus. BM25 covers that gap.

The reranker is *fused* with retrieval rather than overriding it, so no single
stage can drop a correct result.

| Configuration | hit@5 | MRR |
|---|---|---|
| dense only | 90% | 0.812 |
| dense + rerank | 95% | 0.867 |
| hybrid only | 100% | 0.829 |
| **hybrid + fused rerank** | **100%** | **0.875** |

```bash
python scripts/evaluate.py      # reproduce
```

---

## Commands

```bash
# Ask
python scripts/ask.py                                   # interactive
python scripts/ask.py "what caused INC-2024-017?"
python scripts/ask.py --model minimax "..."             # switch LLM
python scripts/ask.py --dept finance "..."              # filter department

# Inspect retrieval (no LLM, instant)
python scripts/ask.py --retrieval-only "..."
python scripts/ask.py --no-rerank "..."
python scripts/ask.py --no-hybrid "..."

# Index
python scripts/index_data.py
python scripts/index_data.py --stats

# Test
python scripts/evaluate.py
python scripts/test_mcp_server.py
```

---

## MCP server

| Tool | Purpose |
|---|---|
| `search_database` | Hybrid search, optional department filter |
| `fetch_document` | Read a full document after search identifies it |
| `list_documents` | Corpus map, so the model stops guessing paths |
| `search_web` | DuckDuckGo, no API key |

Register with Claude Code:

```bash
claude mcp add helix-rag -- python mcp_server/server.py
```

For Claude Desktop, see `mcp_server/claude_desktop_config.example.json`.

Running `python mcp_server/server.py` by hand will look like it hangs — that is
correct. It is a stdio server waiting for JSON-RPC on stdin; a client launches
it.

---

## Configuration

All in `src/config.py`, overridable by environment variable.

| Variable | Default | Purpose |
|---|---|---|
| `HELIX_MODEL` | `deepseek` | Which entry in `MODELS` |
| `HELIX_HYBRID` | `1` | BM25 + dense fusion |
| `HELIX_RERANK` | `1` | Cross-encoder reranking |
| `HELIX_TOP_K` | `5` | Passages passed to the LLM |
| `HELIX_RETRIEVE_K` | `30` | Candidate pool before reranking |
| `HELIX_CHUNK_TOKENS` | `400` | Chunk budget |

```python
MODELS = {
    "deepseek": "deepseek-coder-v2:lite",   # default, local
    "minimax":  "minimax-m2.5:cloud",       # better at prose synthesis
}
```

> `deepseek-coder-v2:lite` is a *code* model. It works, but is weaker at prose
> synthesis and citation discipline than a general model. If answers look wrong,
> check `--retrieval-only` first to tell a retrieval problem from an LLM one.

---

## Notes

- Chunks are 400 tokens, not 500: `bge-small` truncates at 512 and each chunk
  gets a title header prepended at embedding time.
- BM25 does not split compound identifiers — splitting `inc-2024-017` injects
  `2024`, which appears in 375 of 436 chunks and dilutes the term.
- The MCP server redirects `sys.stdout` during startup. Under stdio transport
  stdout is the JSON-RPC channel, and a stray progress bar corrupts it, causing
  the client to hang rather than fail.
