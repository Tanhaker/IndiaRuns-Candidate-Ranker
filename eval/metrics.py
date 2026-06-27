"""
metrics.py — Ranking-quality metrics matching the challenge's scoring rubric.

Final composite (from the submission spec):

    score = 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10

NDCG uses *graded* relevance (tier 0..4). MAP and Precision@k use *binary*
relevance, where a candidate counts as relevant iff their tier >= REL_THRESHOLD.

All functions take:
    ranking  : list of candidate_id, best-first (your submission order)
    rel      : dict candidate_id -> integer relevance tier (your labels)
Candidates in `ranking` but absent from `rel` are treated as tier 0 (unknown =
not-relevant), which is the conservative assumption for an unlabeled item.

Pure stdlib + math. No numpy dependency so the harness runs anywhere.
"""
from __future__ import annotations
import math

REL_THRESHOLD = 3   # tier >= 3 is "relevant" for binary metrics (MAP, P@k)
MAX_TIER = 4        # perfect-fit tier; used to build the ideal DCG


def _gain(tier: int) -> float:
    """Graded gain. 2^tier - 1 rewards top tiers steeply (standard NDCG gain)."""
    return (2 ** tier) - 1


def dcg_at_k(ranking, rel, k):
    dcg = 0.0
    for i, cid in enumerate(ranking[:k]):
        tier = rel.get(cid, 0)
        dcg += _gain(tier) / math.log2(i + 2)  # i+2 because positions are 1-based
    return dcg


def ndcg_at_k(ranking, rel, k):
    """Normalized DCG@k against the ideal ordering of the labeled tiers."""
    dcg = dcg_at_k(ranking, rel, k)
    ideal = sorted(rel.values(), reverse=True)
    idcg = sum(_gain(t) / math.log2(i + 2) for i, t in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranking, rel, k):
    if k == 0:
        return 0.0
    hits = sum(1 for cid in ranking[:k] if rel.get(cid, 0) >= REL_THRESHOLD)
    return hits / k


def average_precision(ranking, rel):
    """AP for a single query (this challenge is effectively one query/JD)."""
    n_rel = sum(1 for t in rel.values() if t >= REL_THRESHOLD)
    if n_rel == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, cid in enumerate(ranking):
        if rel.get(cid, 0) >= REL_THRESHOLD:
            hits += 1
            ap += hits / (i + 1)
    return ap / n_rel


def composite(ranking, rel):
    """The exact weighted blend the challenge uses, plus its components."""
    parts = {
        "ndcg@10": ndcg_at_k(ranking, rel, 10),
        "ndcg@50": ndcg_at_k(ranking, rel, 50),
        "map": average_precision(ranking, rel),
        "p@10": precision_at_k(ranking, rel, 10),
    }
    parts["composite"] = (
        0.50 * parts["ndcg@10"]
        + 0.30 * parts["ndcg@50"]
        + 0.15 * parts["map"]
        + 0.05 * parts["p@10"]
    )
    return parts
