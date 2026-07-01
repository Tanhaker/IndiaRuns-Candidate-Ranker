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


def _bm25_scores(texts, query):
    """
    Okapi BM25 of the JD query against each candidate document — a sparse,
    lexical relevance lane that complements TF-IDF cosine. Built on a CountVectorizer
    so it stays CPU-only and memory-light. Returns raw BM25 scores (higher = better).

    BM25 rewards documents where the JD's rarer terms appear, with saturation on
    term frequency and a length-normalisation that TF-IDF cosine lacks — which is
    why fusing the two (below) tends to beat either alone.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as sp

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    try:
        tf = vec.fit_transform([t.lower() for t in texts])      # docs x terms
    except ValueError:
        return np.zeros(len(texts))
    q_terms = vec.transform([query.lower()])
    q_idx = q_terms.indices
    if q_idx.size == 0:
        return np.zeros(len(texts))

    k1, b = 1.5, 0.75
    dl = np.asarray(tf.sum(axis=1)).ravel()                     # doc lengths
    avgdl = dl.mean() if dl.mean() > 0 else 1.0
    n_docs = tf.shape[0]
    df = np.asarray((tf > 0).sum(axis=0)).ravel()               # doc freq per term
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    scores = np.zeros(n_docs)
    tf_csc = tf.tocsc()
    for t in q_idx:
        col = tf_csc.getcol(t).toarray().ravel()                # tf of term t per doc
        denom = col + k1 * (1 - b + b * dl / avgdl)
        scores += idf[t] * (col * (k1 + 1)) / np.where(denom == 0, 1, denom)
    return scores


def _rrf(*score_arrays, k=60):
    """
    Reciprocal Rank Fusion of several score arrays into one [0,1] signal.
    Each lane votes by RANK, not raw score, so the fusion is scale-free and
    robust — a candidate ranked high by either lane rises. RRF(d) = Σ 1/(k+rank_d).
    """
    n = len(score_arrays[0])
    fused = np.zeros(n)
    for arr in score_arrays:
        order = np.argsort(-arr)              # indices best→worst
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n)            # 0-based rank per doc
        fused += 1.0 / (k + rank + 1.0)
    if fused.max() > fused.min():
        fused = (fused - fused.min()) / (fused.max() - fused.min())
    return fused


def semantic_scores(candidates):
    """
    Hybrid lexical+semantic relevance to the JD, fused by Reciprocal Rank Fusion.

    Two complementary lanes over each candidate's text:
      * TF-IDF cosine  — semantic/term-weighted similarity to the JD query.
      * Okapi BM25     — sparse lexical relevance with length normalisation.
    RRF combines them by rank (scale-free), which the IR literature shows beats
    either lane alone. Returns a normalized array in [0,1] so it composes with
    the other [0,1] FIT components.

    Robust to degenerate input: if the texts yield no usable vocabulary (e.g. an
    uploaded file in the wrong schema), it returns zeros instead of raising, so
    the sandbox never crashes on a malformed upload.
    """
    texts = [F.candidate_text(c) for c in candidates]
    if not any(t and t.strip() for t in texts):
        return np.zeros(len(candidates))

    # --- lane 1: TF-IDF cosine -------------------------------------------
    min_df = 3 if len(texts) >= 8 else 1
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_df=0.6, sublinear_tf=True,
        max_features=60000, stop_words="english",
    )
    try:
        X = vec.fit_transform(texts)
        q = vec.transform([C.JD_QUERY])
        tfidf = linear_kernel(q, X).ravel()
    except ValueError:
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  max_features=60000)
            X = vec.fit_transform(texts)
            tfidf = linear_kernel(vec.transform([C.JD_QUERY]), X).ravel()
        except ValueError:
            return np.zeros(len(candidates))

    # --- lane 2: BM25 lexical --------------------------------------------
    bm25 = _bm25_scores(texts, C.JD_QUERY)

    # --- fuse by rank (RRF) ----------------------------------------------
    if bm25.max() > bm25.min():
        return _rrf(tfidf, bm25)
    # BM25 degenerate (tiny sample) → fall back to TF-IDF alone, normalized.
    if tfidf.max() > tfidf.min():
        return (tfidf - tfidf.min()) / (tfidf.max() - tfidf.min())
    return np.zeros(len(candidates))


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
