"""Query the RAG system directly from the terminal.

    python scripts/ask.py                          # interactive
    python scripts/ask.py "what caused INC-2024-017?"
    python scripts/ask.py --model minimax "..."    # switch LLM
    python scripts/ask.py --dept finance "..."     # filter by department
    python scripts/ask.py --retrieval-only "..."   # skip the LLM, show chunks
    python scripts/ask.py --no-rerank "..."        # stage-1 vector search only

Interactive commands:  :dept <name>   :model <key>   :rerank on|off
                       :sources       :stats        :help        :quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm, rag, vector_store  # noqa: E402

BOLD, DIM, CYAN, YELLOW, GREEN, RED, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m"
)


def format_scores(r: dict) -> str:
    """Render whichever retrieval stages contributed to this hit."""
    parts = []
    if r.get("vector_score") is not None:
        parts.append(f"vec {r['vector_score']:.3f}")
    elif r.get("score") is not None:
        parts.append(f"vec {r['score']:.3f}")
    if r.get("bm25_score") is not None:
        parts.append(f"bm25 {r['bm25_score']:.2f}")
    if r.get("rerank_score") is not None:
        parts.append(f"rerank {r['rerank_score']:+.2f}")
    if r.get("rrf_score") is not None:
        parts.append(f"rrf {r['rrf_score']:.4f}")
    return "  ".join(parts)


def print_sources(sources: list[dict]) -> None:
    if not sources:
        return
    print(f"\n{DIM}{'-' * 70}{RESET}")
    print(f"{DIM}Sources{RESET}")
    for s in sources:
        print(
            f"  {CYAN}[{s['n']}]{RESET} {s['doc_title']}\n"
            f"      {DIM}{s['department']}/{Path(s['source_file']).name}  "
            f"part {s['chunk_index'] + 1}/{s['total_chunks']}  "
            f"{format_scores(s)}{RESET}"
        )


def show_retrieval(question: str, args) -> None:
    """Retrieval only, no LLM. Useful for tuning and for demos."""
    results = rag.retrieve(
        question,
        top_k=args.top_k,
        department=args.dept,
        use_rerank=not args.no_rerank,
        use_hybrid=not args.no_hybrid,
    )
    if not results:
        print(f"{YELLOW}No matches.{RESET}")
        return

    for i, r in enumerate(results, start=1):
        print(f"\n{CYAN}[{i}]{RESET} {BOLD}{r['doc_title']}{RESET}")
        print(f"{DIM}    {r['department']}/{Path(r['source_file']).name}  "
              f"part {int(r['chunk_index']) + 1}/{r['total_chunks']}  "
              f"{format_scores(r)}{RESET}")
        excerpt = r["text"].strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600].rsplit(" ", 1)[0] + " ..."
        for line in excerpt.splitlines():
            print(f"    {line}")


def ask_once(question: str, args) -> None:
    if args.retrieval_only:
        show_retrieval(question, args)
        return

    print()
    sources: list[dict] = []
    first_token = True

    for kind, payload in rag.answer_stream(
        question,
        top_k=args.top_k,
        department=args.dept,
        model_key=args.model,
        use_rerank=not args.no_rerank,
        use_hybrid=not args.no_hybrid,
    ):
        if kind == "sources":
            sources = payload
            if not sources:
                print(f"{YELLOW}Nothing in the corpus matched that question.{RESET}")
                return
            print(f"{DIM}Retrieved {len(sources)} passages. Generating ...{RESET}\n")
        else:
            if first_token:
                first_token = False
            print(payload, end="", flush=True)

    print()
    if args.show_sources:
        print_sources(sources)


HELP = """
Commands
  :dept <name>     filter to one department (:dept all to clear)
  :model <key>     switch LLM
  :rerank on|off   toggle cross-encoder reranking
  :sources on|off  toggle the source list after each answer
  :stats           index statistics
  :help            this message
  :quit            exit
"""


def repl(args) -> None:
    stats = vector_store.stats()
    print(f"\n{BOLD}Helix Robotics RAG{RESET}")
    print(f"{DIM}{stats['documents']} documents, {stats['chunks']} chunks  |  "
          f"model {llm.resolve_model(args.model)}  |  "
          f"hybrid {'on' if not args.no_hybrid else 'off'}  |  "
          f"rerank {'on' if not args.no_rerank else 'off'}{RESET}")
    print(f"{DIM}Ask a question, or :help for commands.{RESET}")

    while True:
        try:
            scope = f" [{args.dept}]" if args.dept else ""
            line = input(f"\n{GREEN}?{scope}{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue

        if line.startswith(":"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "q", "exit"):
                return
            if cmd == "help":
                print(HELP)
            elif cmd == "dept":
                if arg in ("all", "none", ""):
                    args.dept = None
                    print(f"{DIM}Department filter cleared.{RESET}")
                elif arg in config.DEPARTMENTS:
                    args.dept = arg
                    print(f"{DIM}Filtering to {arg}.{RESET}")
                else:
                    print(f"{RED}Unknown department. "
                          f"One of: {', '.join(config.DEPARTMENTS)}{RESET}")
            elif cmd == "model":
                if arg in config.MODELS:
                    args.model = arg
                    print(f"{DIM}Model set to {config.MODELS[arg]}.{RESET}")
                else:
                    print(f"{RED}Unknown model. "
                          f"One of: {', '.join(sorted(config.MODELS))}{RESET}")
            elif cmd == "rerank":
                args.no_rerank = arg == "off"
                print(f"{DIM}Rerank {'off' if args.no_rerank else 'on'}.{RESET}")
            elif cmd == "sources":
                args.show_sources = arg != "off"
                print(f"{DIM}Sources {'on' if args.show_sources else 'off'}.{RESET}")
            elif cmd == "stats":
                s = vector_store.stats()
                print(f"\n  {s['documents']} documents, {s['chunks']} chunks")
                for dept, e in s["by_department"].items():
                    print(f"    {dept:<14} {e['documents']:>3} docs  "
                          f"{e['chunks']:>5} chunks")
            else:
                print(f"{RED}Unknown command. :help for the list.{RESET}")
            continue

        try:
            ask_once(line, args)
        except Exception as exc:  # keep the REPL alive
            print(f"\n{RED}Error: {exc}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask questions of the Helix Robotics corpus."
    )
    parser.add_argument("question", nargs="*", help="question (omit for interactive)")
    parser.add_argument("--model", default=None,
                        help=f"one of: {', '.join(sorted(config.MODELS))}")
    parser.add_argument("--dept", default=None, choices=config.DEPARTMENTS,
                        help="restrict retrieval to one department")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"passages to pass to the LLM (default {config.TOP_K})")
    parser.add_argument("--no-rerank", action="store_true",
                        help="skip cross-encoder reranking")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="dense vector search only, no BM25 fusion")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="show retrieved chunks without calling the LLM")
    parser.add_argument("--no-sources", dest="show_sources", action="store_false",
                        help="hide the source list")
    parser.set_defaults(show_sources=True)
    args = parser.parse_args()

    if vector_store.count() == 0:
        print(f"{RED}Index is empty.{RESET} Run: python scripts/index_data.py",
              file=sys.stderr)
        return 1

    if not args.retrieval_only:
        ready, message = llm.check_ready(args.model)
        if not ready:
            print(f"{RED}{message}{RESET}", file=sys.stderr)
            return 1

    if args.question:
        ask_once(" ".join(args.question), args)
        return 0

    repl(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
