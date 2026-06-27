"""
pipeline.py — End-to-end ranking pipeline. CPU-only, no network.

Steps:
  1. Stream-load candidates.jsonl (handles .jsonl or .jsonl.gz).
  2. Build a TF-IDF index over candidate text + the JD query; cosine = semantic lane.
  3. Score every candidate (rules + corroboration + semantic + behavioral + guards).
  4. Rank, take top-100, generate reasoning, write a spec-valid CSV.

Designed to finish well within the 5-minute / 16 GB / CPU budget on 100K rows.
"""
from __future__ import annotations
import gzip
import json
import csv
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config as C
from . import features as F
from . import score as S
from . import reasoning as R


def load_candidates(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def semantic_scores(candidates):
    """
    TF-IDF cosine between the JD query and each candidate's text.
    Word 1-2 grams, sublinear tf, English stop-words. Returns a normalized
    array in [0,1] (per-pool min-max) so it composes with the other [0,1]
    components. Fast: ~vectorize once, one matrix-vector product.
    """
    texts = [F.candidate_text(c) for c in candidates]
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True,
        max_features=60000, stop_words="english",
    )
    X = vec.fit_transform(texts)
    q = vec.transform([C.JD_QUERY])
    sims = linear_kernel(q, X).ravel()
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    return sims


def rank(candidates, top_n=C.TOP_N, verbose=True):
    t0 = time.time()
    if verbose:
        print(f"[1/4] loaded {len(candidates):,} candidates")

    sims = semantic_scores(candidates)
    if verbose:
        print(f"[2/4] semantic index built ({time.time()-t0:.1f}s)")

    scored = [S.score_candidate(c, float(sims[i])) for i, c in enumerate(candidates)]
    if verbose:
        print(f"[3/4] scored all candidates ({time.time()-t0:.1f}s)")

    by_id = {c["candidate_id"]: c for c in candidates}
    # Sort: final score desc, then candidate_id asc (deterministic spec tie-break).
    scored.sort(key=lambda r: (-r["final"], r["candidate_id"]))
    top = scored[:top_n]

    rows = []
    prev = None
    for i, sc in enumerate(top):
        raw = round(sc["final"], 6)
        # Guarantee strictly non-increasing written score even after rounding.
        s = raw if prev is None else min(raw, prev)
        prev = s
        reason = R.build_reasoning(by_id[sc["candidate_id"]], sc)
        rows.append({
            "candidate_id": sc["candidate_id"],
            "rank": i + 1,
            "score": f"{s:.6f}",
            "reasoning": reason,
        })
    if verbose:
        hp = sum(1 for sc in top if sc["honeypot"])
        print(f"[4/4] top-{top_n} selected; honeypots in top-{top_n}: {hp} "
              f"({time.time()-t0:.1f}s total)")
    return rows, scored


def write_csv(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
