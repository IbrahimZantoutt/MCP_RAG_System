"""Retrieval evaluation harness.

Measures whether the right documents come back, across retrieval
configurations. Retrieval only -- no LLM involved, so it runs in seconds and
its results are deterministic.

    python scripts/evaluate.py              # compare all configurations
    python scripts/evaluate.py --verbose    # show per-question outcomes
    python scripts/evaluate.py --k 5

Metrics:
    hit@k   fraction of questions where at least one expected document appears
            in the top k results
    MRR     mean reciprocal rank of the first expected document (1.0 = always
            ranked first, 0.5 = typically second, 0 = never found)

The eval set is deliberately mixed: questions answerable from one document,
questions needing several, exact-identifier lookups, and paraphrased questions
that share no vocabulary with their source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag, vector_store  # noqa: E402

# (question, [documents that would count as correct])
EVAL_SET: list[tuple[str, list[str]]] = [
    # -- exact identifier lookups -------------------------------------------
    ("What caused incident INC-2024-017 and what did it cost the company in total?",
     ["engineering/postmortem_INC-2024-017.txt"]),
    ("What does fault code E-611 mean and how do I resolve it?",
     ["support/troubleshooting_guide.txt"]),
    ("Which firmware release fixed the state of charge hysteresis problem?",
     ["engineering/firmware_release_notes.txt"]),
    ("What changed in Fleet OS 4.2.2?",
     ["product/fleet_os_4.2_release.txt", "engineering/postmortem_INC-2024-017.txt"]),
    ("Which ISO standard covers driverless industrial trucks?",
     ["compliance/safety_standards.txt"]),

    # -- single-document factual --------------------------------------------
    ("How much parental leave do employees get?",
     ["hr/employee_handbook.txt"]),
    ("What is the recommended charging dock ratio for the HX-450 Stacker?",
     ["product/hx450_stacker_spec.txt", "support/customer_faq.txt"]),
    ("What is the gross margin on an HX-200 base unit?",
     ["finance/unit_economics_hx200.txt"]),
    ("What are the on-call standby payment rates?",
     ["hr/oncall_compensation_policy.txt"]),
    ("How much discount can the VP of Sales approve?",
     ["sales/pricing_and_packaging.txt"]),
    ("What is the maximum speed of the HX-450 with the mast extended?",
     ["product/hx450_stacker_spec.txt", "compliance/safety_standards.txt"]),

    # -- procedural ----------------------------------------------------------
    ("How do I recover a fleet where robots are stuck holding dock reservations?",
     ["engineering/runbook_fleet_recovery.txt", "support/troubleshooting_guide.txt"]),
    ("What is the floor entry protocol before walking into a robot area?",
     ["hr/field_technician_training.txt"]),

    # -- paraphrased: little vocabulary overlap with the source --------------
    ("Why can a software bug never make a robot move dangerously?",
     ["compliance/safety_standards.txt", "engineering/fleet_os_architecture.txt",
      "product/hx200_courier_spec.txt"]),
    ("What is stopping us from selling more to the Swiss pharmaceutical customer?",
     ["sales/account_voss_pharma.txt"]),
    ("Why is our profitability worse than it should be?",
     ["finance/q3_2024_financial_review.txt", "finance/unit_economics_hx200.txt"]),

    # -- multi-document synthesis -------------------------------------------
    ("How large was the service credit paid to Cardinal Foods and why was it "
     "larger than the contract required?",
     ["sales/account_cardinal_foods.txt", "support/sla_and_escalation.txt",
      "finance/q3_2024_financial_review.txt"]),
    ("Why did Helix change how it measures fleet availability?",
     ["support/sla_and_escalation.txt", "sales/account_cardinal_foods.txt",
      "finance/q3_2024_financial_review.txt"]),
    ("What is blocked on single sign-on?",
     ["sales/account_voss_pharma.txt", "product/product_roadmap_2025.txt",
      "compliance/data_privacy_policy.txt", "sales/pricing_and_packaging.txt"]),
    ("What did the company change about on-call after the September outage?",
     ["hr/oncall_compensation_policy.txt", "finance/travel_expense_policy.txt"]),
]

CONFIGS = {
    "dense only":        dict(use_hybrid=False, use_rerank=False),
    "dense + rerank":    dict(use_hybrid=False, use_rerank=True),
    "hybrid only":       dict(use_hybrid=True, use_rerank=False),
    "hybrid + rerank":   dict(use_hybrid=True, use_rerank=True),
}


def evaluate(config_kwargs: dict, k: int, verbose: bool = False) -> dict:
    hits = 0
    reciprocal_ranks = []
    failures = []

    for question, expected in EVAL_SET:
        results = rag.retrieve(question, top_k=k, **config_kwargs)
        found = [r["source_file"] for r in results]

        rank = None
        for i, source in enumerate(found, start=1):
            if source in expected:
                rank = i
                break

        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append((question, expected, found))

        if verbose:
            mark = f"rank {rank}" if rank else "MISS"
            print(f"  [{mark:>7}] {question[:66]}")

    n = len(EVAL_SET)
    return {
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "hits": hits,
        "total": n,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--k", type=int, default=5, help="results per query")
    parser.add_argument("--verbose", action="store_true",
                        help="show per-question ranks")
    parser.add_argument("--failures", action="store_true",
                        help="show what was retrieved for misses")
    args = parser.parse_args()

    if vector_store.count() == 0:
        print("Index is empty. Run: python scripts/index_data.py", file=sys.stderr)
        return 1

    print(f"\nEvaluating {len(EVAL_SET)} questions at k={args.k}\n")

    results = {}
    for name, kwargs in CONFIGS.items():
        if args.verbose:
            print(f"{name}:")
        results[name] = evaluate(kwargs, args.k, args.verbose)
        if args.verbose:
            print()

    print(f"{'configuration':<20} {'hit@' + str(args.k):>8} {'MRR':>8}")
    print("-" * 38)
    best = max(results.values(), key=lambda r: (r["mrr"], r["hit_rate"]))
    for name, r in results.items():
        marker = "  <-- best" if r is best else ""
        print(f"{name:<20} {r['hit_rate']:>7.0%} {r['mrr']:>8.3f}{marker}")
    print()

    if args.failures:
        for name, r in results.items():
            if not r["failures"]:
                continue
            print(f"\nMisses for {name}:")
            for question, expected, found in r["failures"]:
                print(f"\n  Q: {question}")
                print(f"     expected one of: {', '.join(expected)}")
                print("     got:")
                for f in found:
                    print(f"       - {f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
