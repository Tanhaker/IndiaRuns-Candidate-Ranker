"""
make_labels.py — Build a labeling worksheet from the public sample.

You cannot see the hidden ground truth, so to measure your ranker you label a
small validation set yourself. This script reads the 50-candidate public sample
and writes a CSV worksheet with:

    candidate_id, suggested_tier, your_tier, current_title, yoe, why

`suggested_tier` is a heuristic first guess so you are correcting, not labeling
from a blank page — but the WHOLE POINT is that you override it with your own
judgement in `your_tier`. The heuristic is intentionally simple and must not be
trusted blindly; if it were trustworthy you would not need an eval set.

Tier scale (graded relevance):
    4  perfect-fit senior AI/ML/retrieval engineer, product co, corroborated
    3  strong, clearly relevant, minor gaps
    2  plausible / adjacent (e.g. backend leaning ML, or strong but off-band)
    1  weak — keyword overlap only, or wrong specialism
    0  irrelevant OR honeypot (impossible profile) OR pure keyword-stuffer

Usage:
    python -m eval.make_labels --sample sample_candidates.json --out eval/labels.csv
Then open eval/labels.csv and fill the `your_tier` column by hand.
"""
from __future__ import annotations
import argparse
import csv
import json

STRONG_TITLES = ("ai engineer", "machine learning engineer", "ml engineer",
                 "applied scientist", "research engineer", "nlp engineer",
                 "recommendation", "search engineer", "ranking")
RETRIEVAL = ("retrieval", "embedding", "vector", "faiss", "ranking", "rerank",
             "recommendation", "semantic search", "ndcg", "mrr")
NONTECH = ("manager", "marketing", "hr", "accountant", "sales", "mechanical",
           "civil", "recruiter", "business analyst", "teacher", "nurse")
PRODUCT_COS = ("razorpay", "zomato", "swiggy", "cred", "flipkart", "uber",
               "google", "meta", "apple", "netflix", "linkedin", "microsoft",
               "amazon", "paytm", "phonepe", "ola", "myntra")


def _suggest_tier(c):
    prof = c.get("profile", {}) or {}
    title = (prof.get("current_title") or "").lower()
    text = " ".join(
        (h.get("description") or "") for h in c.get("career_history", [])
    ).lower()
    companies = " ".join(
        (h.get("company") or "") for h in c.get("career_history", [])
    ).lower()
    yoe = prof.get("years_of_experience", 0) or 0

    if any(k in title for k in NONTECH):
        return 0  # stuffer host-body title
    strong_title = any(k in title for k in STRONG_TITLES)
    has_retrieval = any(k in text for k in RETRIEVAL)
    at_product = any(k in companies for k in PRODUCT_COS)

    if strong_title and has_retrieval and at_product and 4 <= yoe <= 10:
        return 4
    if strong_title and (has_retrieval or at_product):
        return 3
    if strong_title or has_retrieval:
        return 2
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sample_candidates.json")
    ap.add_argument("--out", default="eval/labels.csv")
    args = ap.parse_args()

    with open(args.sample, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for c in data:
        prof = c.get("profile", {}) or {}
        rows.append({
            "candidate_id": c["candidate_id"],
            "suggested_tier": _suggest_tier(c),
            "your_tier": "",  # <- you fill this in
            "current_title": prof.get("current_title", ""),
            "yoe": prof.get("years_of_experience", ""),
            "why": "",  # optional note to yourself
        })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print("Now open it and fill the `your_tier` column (0-4) using your judgement.")
    print("`suggested_tier` is only a starting hint — override it freely.")


if __name__ == "__main__":
    main()
