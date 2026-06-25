#!/usr/bin/env python3
"""
rank.py — Single-command reproduction of the submission CSV.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

CPU-only, no network. Finishes within the 5-minute / 16 GB budget on 100K rows.
"""
import argparse
import time
from redrob_ranker import pipeline


def main():
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD.")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl (or .jsonl.gz).")
    ap.add_argument("--out", default="submission.csv",
                    help="Output CSV path (default: submission.csv).")
    ap.add_argument("--top", type=int, default=100, help="How many to rank (default 100).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    candidates = pipeline.load_candidates(args.candidates)
    rows, _ = pipeline.rank(candidates, top_n=args.top, verbose=not args.quiet)
    pipeline.write_csv(rows, args.out)
    print(f"Wrote {len(rows)} ranked candidates to {args.out} in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
