#!/usr/bin/env python3
"""Redrob candidate ranker - command-line entry point.

Produces the top-100 submission CSV for the "Senior AI Engineer - Founding Team"
job description from a candidates JSONL/JSONL.gz file.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Design goals (see README.md):
* Fully offline, CPU-only, deterministic, <= 5 minutes for 100K candidates.
* Hybrid scoring: semantic match (TF-IDF) + interpretable rule components +
  behavioral availability modifier + explicit JD disqualifiers + honeypot guard.
"""
from __future__ import annotations

import argparse
import sys
import time

from ranker.io_utils import load_candidates, write_submission
from ranker.reasoning import build_reasoning
from ranker.scoring import score_candidate
from ranker.semantic import build_semantic
from ranker.text import full_text


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Rank candidates for the Redrob AI JD.")
    p.add_argument("--candidates", required=True,
                   help="Path to candidates.jsonl or candidates.jsonl.gz")
    p.add_argument("--out", default="submission.csv",
                   help="Output CSV path (default: submission.csv)")
    p.add_argument("--top", type=int, default=100,
                   help="Number of candidates to output (default: 100)")
    p.add_argument("--semantic", choices=["tfidf", "st"], default="tfidf",
                   help="Semantic engine: tfidf (default, offline) or st "
                        "(sentence-transformers, needs cached model)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    print(f"[1/5] Loading candidates from {args.candidates} ...", flush=True)
    candidates = load_candidates(args.candidates)
    n = len(candidates)
    print(f"      loaded {n:,} candidates ({time.time() - t0:.1f}s)", flush=True)

    print(f"[2/5] Building semantic index ({args.semantic}) ...", flush=True)
    docs = [full_text(c) for c in candidates]
    sem = build_semantic(args.semantic)
    sem.fit_transform(docs)
    sims = sem.similarities()
    print(f"      semantic index ready ({time.time() - t0:.1f}s)", flush=True)

    print("[3/5] Scoring candidates ...", flush=True)
    results = []
    honeypots = 0
    for cand, sim in zip(candidates, sims):
        res = score_candidate(cand, float(sim))
        if res["honeypot_reasons"]:
            honeypots += 1
        results.append((cand, res))
    print(f"      scored {n:,}; flagged {honeypots} honeypots "
          f"({time.time() - t0:.1f}s)", flush=True)

    print("[4/5] Ranking ...", flush=True)
    # Sort by the *rounded* score (what we actually write) desc, then
    # candidate_id asc. Sorting on the rounded value guarantees the validator's
    # tie-break rule (equal scores -> candidate_id ascending) always holds.
    for _, res in results:
        res["round_score"] = round(res["score"], 4)
    results.sort(key=lambda cr: (-cr[1]["round_score"], cr[0]["candidate_id"]))
    top = results[: args.top]

    print("[5/5] Writing submission ...", flush=True)
    rows = []
    for rank, (cand, res) in enumerate(top, start=1):
        rows.append({
            "candidate_id": cand["candidate_id"],
            "rank": rank,
            "score": res["round_score"],
            "reasoning": build_reasoning(cand, res, rank),
        })
    write_submission(rows, args.out)

    print(f"\nDone. Wrote {len(rows)} rows to {args.out} "
          f"in {time.time() - t0:.1f}s.", flush=True)
    # Quick sanity: report top-5 for a human glance.
    print("\nTop 5 preview:")
    for r in rows[:5]:
        print(f"  {r['rank']:>3}  {r['candidate_id']}  {r['score']:.4f}  "
              f"{r['reasoning'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
