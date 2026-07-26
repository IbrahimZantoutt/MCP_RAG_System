# Helix RAG + MCP Server

A local retrieval-augmented generation system over a synthetic corporate corpus,
exposed both as a command-line tool and as a Model Context Protocol server.

Everything runs locally: embeddings, vector store, reranking, and the LLM.
No API keys, no cloud services, no network required except the optional
`search_web` tool.

```
                      ┌──────────────────┐
   scripts/ask.py ───▶│                  │
                      │     src/rag.py   │──▶ retrieve ──▶ Ollama ──▶ answer
   MCP client    ───▶ │   (one engine)   │
   (Claude, etc.)     └──────────────────┘
```

`ask.py` and the MCP server are two front doors onto one engine, not two
implementations of the same idea.

---

## Quick start

```bash
pip install -r requirements.txt
ollama pull deepseek-coder-v2:lite

python scripts/index_data.py          # build the index  (~40s)
python scripts/ask.py                 # interactive
```

First run downloads two models (~220 MB total): `bge-small-en-v1.5` for
embeddings and `ms-marco-MiniLM-L-6-v2` for reranking.

---

## The corpus

27 documents, ~76,000 words, covering a fictional warehouse-robotics company
called **Helix Robotics** — products, incidents, support tickets, sales
accounts, finances, HR policy, and safety compliance.

| Department | Docs | Contents |
|---|---|---|
| `engineering/` | 5 | Architecture, incident postmortem, recovery runbook, firmware notes, onboarding |
| `support/` | 5 | SLA policy, troubleshooting guide, customer FAQ, 772 support tickets |
| `product/` | 4 | HX-200 and HX-450 robot specs, Fleet OS release notes, 2025 roadmap |
| `sales/` | 4 | Two account records, pricing, competitive landscape |
| `hr/` | 4 | Handbook, on-call compensation, leveling guide, field technician manual |
| `finance/` | 3 | Quarterly financials, unit economics, travel policy |
| `compliance/` | 2 | Safety certification, data privacy |

The documents are **deliberately cross-referenced**. A single incident —
`INC-2024-017`, a charging-dock deadlock that took down 61 robots at the
company's largest customer — is described from seven different angles:

- the **postmortem** has the root cause
- the **runbook** has the recovery procedure
- the **support tickets** have the customer phone call that first reported it
- the **account record** has the commercial fallout
- the **financial review** has the $338,000 cost
- the **SLA policy** has the contract terms that changed because of it
- the **on-call policy** has the compensation rules that changed because of it

No single document tells the whole story, which is the point: it makes retrieval
quality visible rather than theoretical.

Regenerate the ticket logs (deterministic, seeded) with:

```bash
python scripts/generate_ticket_logs.py
```

---

## How retrieval works

```
query
  ├─▶ dense search   (bge-small embeddings, cosine)  ─┐
  │                                                    ├─▶ RRF ─▶ candidates
  └─▶ lexical search (BM25 over the same chunks)     ─┘
                                                            │
                                              cross-encoder rerank
                                                            │
                                            RRF (retrieval + rerank)
                                                            │
                                                        top 5 ─▶ LLM
```

**Why hybrid.** This corpus is saturated with exact identifiers — `INC-2024-017`,
`E-401`, `firmware 3.8.1`, `TKT-8841`. Dense embeddings represent those poorly.
Cosine scores across the whole corpus span only 0.61–0.67, so dense retrieval
alone barely discriminates. BM25 handles exact terms natively.

**Why fuse the reranker instead of letting it decide.** In an earlier
reranker-takes-all design, the cross-encoder took the INC-2024-017 postmortem
from vector rank 2 down to rank 7 — out of the results entirely — because the
query said *"what did it cost"* and the reranker preferred passages about money.
Fusing the two rankings means the reranker can promote and demote, but cannot
single-handedly veto a strong retrieval hit.

Measured on the 20-question eval set:

| configuration | hit@5 | MRR |
|---|---|---|
| dense only | 90% | 0.812 |
| dense + rerank | 95% | 0.867 |
| hybrid only | 100% | 0.829 |
| **hybrid + fused rerank** | **100%** | **0.875** |

```bash
python scripts/evaluate.py             # reproduce the table
python scripts/evaluate.py --failures  # show what misses retrieved instead
```

---

## Command line

```bash
python scripts/ask.py "what caused INC-2024-017?"
python scripts/ask.py --model minimax "why is gross margin low?"
python scripts/ask.py --dept finance "what is the HX-200 bill of materials?"

python scripts/ask.py --retrieval-only "..."   # show chunks, skip the LLM
python scripts/ask.py --no-rerank "..."        # compare retrieval stages
python scripts/ask.py --no-hybrid "..."        # dense only
```

Interactive mode supports `:dept`, `:model`, `:rerank on|off`, `:stats`, `:help`.

Try these — each needs several documents combined:

```
Why did Helix change how it measures fleet availability?
Is Voss Pharma exposed to the charging dock deadlock?
What is blocked on single sign-on, and what is it worth?
What did the company change about on-call after the September outage?
```

The Voss Pharma one is the best demonstration: the answer is **no**, and it is
stated in no document. It requires chaining three facts across three files —
Basel runs the 3.6.x LTS firmware line (account record), that line never adopted
the hysteresis band (firmware notes), and the bug required 3.8.1 (postmortem).

---

## MCP server

```bash
python mcp_server/server.py     # stdio transport
python scripts/test_mcp_server.py   # drive it as a real client
```

| Tool | Purpose |
|---|---|
| `search_database` | Hybrid search over the corpus. Optional department filter. |
| `fetch_document` | Read a full document once search has identified it. |
| `list_documents` | Map of the corpus, so the model stops guessing at paths. |
| `search_web` | DuckDuckGo. No API key. For anything not internal. |

Register it with a client using
[`mcp_server/claude_desktop_config.example.json`](mcp_server/claude_desktop_config.example.json),
or for Claude Code:

```bash
claude mcp add helix-rag -- python mcp_server/server.py
```

**`search_database` returns passages, not a finished answer.** The MCP client is
already a language model; generating an answer inside the tool would mean
running a second, weaker model whose summary discards detail the caller needs.
Retrieval is the tool's job, synthesis is the caller's. `scripts/ask.py` is the
path that does both, for standalone use.

**`fetch_document` is what makes the tool set worth having.** Search returns a
~400-token window; when that window is clearly the right document but cut off,
the model can pull the whole thing itself. Watching a client chain
`search_database` → `fetch_document` unprompted is the most convincing thing
here.

---

## Configuration

Everything lives in [`src/config.py`](src/config.py), overridable by environment
variable.

```python
MODELS = {
    "deepseek": "deepseek-coder-v2:lite",   # default, local
    "minimax":  "minimax-m2.5:cloud",       # stronger at prose synthesis
}
```

| Variable | Default | Purpose |
|---|---|---|
| `HELIX_MODEL` | `deepseek` | Which entry in `MODELS` to use |
| `HELIX_HYBRID` | `1` | BM25 + dense fusion |
| `HELIX_RERANK` | `1` | Cross-encoder reranking |
| `HELIX_TOP_K` | `5` | Passages passed to the LLM |
| `HELIX_RETRIEVE_K` | `30` | Candidate pool before reranking |
| `HELIX_CHUNK_TOKENS` | `400` | Chunk budget |
| `HELIX_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |

> **Note on the default LLM.** `deepseek-coder-v2:lite` is a *code* model. It
> works, but it is noticeably weaker at prose synthesis and citation discipline
> than a general model. If answers feel flat or citations look wrong, that is
> usually the cause rather than retrieval — check with `--retrieval-only` first.
> `--model minimax` or a general model like `qwen3:8b` will do better.

---

## Layout

```
data/                  27 .txt documents, department per directory
src/
  config.py            all tunables
  chunker.py           loading, title extraction, paragraph-aware chunking
  embedder.py          sentence-transformers wrapper
  lexical.py           BM25, implemented directly (no extra dependency)
  fusion.py            reciprocal rank fusion
  reranker.py          cross-encoder
  vector_store.py      ChromaDB persistence
  llm.py               Ollama client and model switching
  rag.py               the engine: retrieve → prompt → grounded answer
scripts/
  generate_ticket_logs.py   corpus generator for the ticket logs
  index_data.py             indexing phase
  ask.py                    interactive CLI
  evaluate.py               retrieval eval harness
  test_mcp_server.py        MCP protocol smoke test
mcp_server/
  server.py            FastMCP stdio server, 4 tools
storage/chroma/        persisted index (gitignored)
```

---

## Implementation notes

Things that turned out to matter, recorded because they were not obvious:

**Chunk budget is 400 tokens, not 500.** `bge-small` truncates at 512. Each
chunk gets a contextual header (`"<title> [<department>]"`) prepended at
embedding time so a mid-document chunk still carries the signal of where it came
from — and 500 + header would silently overflow.

**BM25 does not split compound identifiers.** Splitting `inc-2024-017` into
`["inc", "2024", "017"]` looks helpful, but `2024` appears in 375 of 436 chunks
and dilutes the exact term it was meant to reinforce.

**Stopwords are removed before BM25.** Unfiltered, `"what"` scored idf 2.046 on
this corpus while `"inc-2024-017"` scored 1.769 — queries were partly ranking on
their own question words.

**The identifier is not as discriminative as it looks.** `INC-2024-017` appears
in 74 of 436 chunks precisely because the corpus cross-references it heavily.
The property that makes the demo good is the same one that weakens lexical
retrieval for it.

**The MCP server protects stdout.** Under stdio transport, stdout *is* the
JSON-RPC channel. A tqdm progress bar from sentence-transformers lands there and
corrupts the stream — the client then hangs forever rather than failing, which
is a genuinely annoying thing to debug. `sys.stdout` is pointed at stderr during
import and warmup, progress bars are pinned off, and the real stdout is handed
back only for `mcp.run()`.

**Models are loaded at server startup, not on first call.** Otherwise the first
`search_database` pays a ~10-second model load and looks broken.
