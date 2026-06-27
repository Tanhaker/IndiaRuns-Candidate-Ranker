"""Smoke tests: the discriminations the JD cares about must hold."""
import json, subprocess, sys, os
from redrob_ranker import pipeline, score, features

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "..", "sample_candidates.json")


def _load_sample():
    raw = open(SAMPLE, encoding="utf-8").read().strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]


def test_pipeline_runs_and_csv_valid(tmp_path):
    cands = _load_sample()
    rows, _ = pipeline.rank(cands, top_n=min(25, len(cands)), verbose=False)
    assert len(rows) == min(25, len(cands))
    # ranks unique 1..N, scores non-increasing
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    scores = [float(r["score"]) for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    # reasonings non-empty and varied
    reasons = [r["reasoning"] for r in rows]
    assert all(reasons) and len(set(reasons)) == len(reasons)


def test_corroboration_beats_stuffing():
    """A skill with long duration + endorsements must out-trust a bare keyword."""
    real = {"skills": [{"name": "Embeddings", "proficiency": "expert",
                         "duration_months": 60, "endorsements": 40}]}
    stuffer = {"skills": [{"name": "Embeddings", "proficiency": "beginner",
                           "duration_months": 1, "endorsements": 0}]}
    assert features.skill_features(real)["core"] > 5 * features.skill_features(stuffer)["core"]


def test_nontech_title_floors():
    assert features.title_strength({"profile": {"current_title": "HR Manager"},
                                    "career_history": []}) < 0.1
    assert features.title_strength({"profile": {"current_title": "Senior AI Engineer"},
                                    "career_history": []}) > 0.9


def test_timeline_honeypot_catches_impossible_profile():
    """A profile with overlapping full-time roles must be flagged impossible."""
    from redrob_ranker import honeypot as H
    impossible = {
        "candidate_id": "CAND_TEST_HP",
        "profile": {"current_title": "Senior ML Engineer", "years_of_experience": 6},
        "career_history": [
            {"title": "Senior ML Engineer", "company": "X", "is_current": True,
             "start_date": "2021-01-01", "end_date": None, "duration_months": 54},
            {"title": "ML Engineer", "company": "Y", "is_current": False,
             "start_date": "2020-01-01", "end_date": "2023-06-01", "duration_months": 41},
        ],
        "education": [], "skills": [],
    }
    flags = H.timeline_flags(impossible)
    assert flags, "overlapping full-time roles should be flagged"


def test_timeline_honeypot_clears_normal_profile():
    """A clean sequential career must NOT be flagged (no false positives)."""
    from redrob_ranker import honeypot as H
    clean = {
        "candidate_id": "CAND_TEST_OK",
        "profile": {"current_title": "Senior AI Engineer", "years_of_experience": 6},
        "career_history": [
            {"title": "Senior AI Engineer", "company": "X", "is_current": True,
             "start_date": "2022-01-01", "end_date": None, "duration_months": 29},
            {"title": "ML Engineer", "company": "Y", "is_current": False,
             "start_date": "2019-06-01", "end_date": "2021-12-01", "duration_months": 30},
        ],
        "education": [], "skills": [],
    }
    assert H.timeline_flags(clean) == [], "a normal career must not be flagged"
