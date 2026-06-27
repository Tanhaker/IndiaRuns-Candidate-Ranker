"""
evaluate.py — Score a ranking against your hand labels.

Runs the ranker over the SAME sample you labeled, then measures the produced
order with the challenge's exact composite. Use it as your private leaderboard:
change something in the ranker, re-run this, and see whether the number moved
BEFORE you spend one of your 3 real submissions.

Two ways to provide the ranking being judged:
  (default) re-run the live ranker over the sample, so you measure current code;
  --ranking PATH   score a pre-made submission CSV instead (must use the sample).

Usage:
    python -m eval.make_labels                 # once, then hand-fill labels.csv
    python -m eval.evaluate                     # measure current ranker
    python -m eval.evaluate --ranking my.csv    # measure a saved CSV
"""
from __future__ import annotations
import argparse
import csv
import json
import sys

from eval import metrics as M


def load_labels(path):
    rel = {}
    n_blank = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("your_tier") or "").strip()
            if raw == "":
                # fall back to the suggestion so a partially-labeled file still
                # runs, but warn — unlabeled rows weaken the signal.
                raw = (row.get("suggested_tier") or "0").strip()
                n_blank += 1
            try:
                rel[row["candidate_id"]] = int(float(raw))
            except ValueError:
                rel[row["candidate_id"]] = 0
    return rel, n_blank


def ranking_from_live(sample_path):
    """Run the current ranker over the sample and return ids best-first."""
    from redrob_ranker import pipeline as P
    cands = json.load(open(sample_path, encoding="utf-8"))
    sims = P.semantic_scores(cands)
    from redrob_ranker import score as S
    scored = [S.score_candidate(c, float(sims[i])) for i, c in enumerate(cands)]
    scored.sort(key=lambda r: (-r["final"], r["candidate_id"]))
    return [s["candidate_id"] for s in scored]


def ranking_from_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["rank"]), row["candidate_id"]))
    rows.sort()
    return [cid for _, cid in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/labels.csv")
    ap.add_argument("--sample", default="sample_candidates.json")
    ap.add_argument("--ranking", default=None,
                    help="score a saved submission CSV instead of the live ranker")
    args = ap.parse_args()

    try:
        rel, n_blank = load_labels(args.labels)
    except FileNotFoundError:
        sys.exit(f"No labels at {args.labels}. Run:  python -m eval.make_labels  first.")

    if args.ranking:
        ranking = ranking_from_csv(args.ranking)
        source = f"saved CSV {args.ranking}"
    else:
        ranking = ranking_from_live(args.sample)
        source = "live ranker over sample"

    parts = M.composite(ranking, rel)
    n_rel = sum(1 for t in rel.values() if t >= M.REL_THRESHOLD)

    print(f"\n  Eval set: {len(rel)} labeled candidates "
          f"({n_rel} relevant @ tier>={M.REL_THRESHOLD})")
    if n_blank:
        print(f"  ! {n_blank} rows had no your_tier — fell back to suggested_tier. "
              f"Fill them for a trustworthy number.")
    print(f"  Ranking source: {source}\n")
    print(f"    NDCG@10 : {parts['ndcg@10']:.4f}   (weight 0.50)")
    print(f"    NDCG@50 : {parts['ndcg@50']:.4f}   (weight 0.30)")
    print(f"    MAP     : {parts['map']:.4f}   (weight 0.15)")
    print(f"    P@10    : {parts['p@10']:.4f}   (weight 0.05)")
    print(f"    " + "-" * 28)
    print(f"    COMPOSITE: {parts['composite']:.4f}\n")


if __name__ == "__main__":
    main()
