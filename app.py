"""
app.py — Sandbox demo (submission_spec Section 10.5).

Accepts a small candidate sample (≤100, uploaded .jsonl/.json or the bundled
sample_candidates.json), runs the full ranking pipeline, and returns a ranked
CSV — all on CPU within the compute budget.

    streamlit run app.py
"""
import json
import io
import streamlit as st
from redrob_ranker import pipeline

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Senior AI Engineer Candidate Ranker")
st.caption("Hybrid rule-based + TF-IDF semantic ranker. CPU-only, no network, no LLM calls.")

st.markdown(
    "Upload a small candidate sample (`.jsonl` one-object-per-line, or a `.json` "
    "list) — or use the bundled sample — and the system returns a ranked shortlist."
)

# --- Downloadable format template -----------------------------------------
# One fully-formed candidate in the expected schema. Users fill this in to test
# with their own candidates. (JSON, not CSV: a candidate has nested career
# history and per-skill evidence that a flat CSV row cannot hold.)
_TEMPLATE = [{
    "candidate_id": "CAND_0001",
    "profile": {
        "anonymized_name": "Candidate One",
        "headline": "Senior AI Engineer",
        "summary": "Builds retrieval and ranking systems with embeddings and vector search.",
        "location": "Bangalore", "country": "India",
        "years_of_experience": 6,
        "current_title": "Senior AI Engineer", "current_company": "Acme",
        "current_company_size": "1000-5000", "current_industry": "Technology"
    },
    "career_history": [
        {"title": "Senior AI Engineer", "company": "Acme", "is_current": True,
         "start_date": "2022-01-01", "end_date": None, "duration_months": 30,
         "description": "Built a retrieval/ranking pipeline with embeddings, FAISS, NDCG eval."},
        {"title": "ML Engineer", "company": "Globex", "is_current": False,
         "start_date": "2019-06-01", "end_date": "2021-12-01", "duration_months": 30,
         "description": "Recommendation ranking and semantic search in production."}
    ],
    "education": [{"degree": "B.Tech Computer Science", "institution": "Some University", "end_year": 2017}],
    "skills": [
        {"name": "Information Retrieval", "proficiency": "expert", "duration_months": 48, "endorsements": 25},
        {"name": "Embeddings", "proficiency": "advanced", "duration_months": 36, "endorsements": 18}
    ],
    "certifications": [], "languages": ["English"],
    "redrob_signals": {
        "open_to_work_flag": True, "recruiter_response_rate": 0.8,
        "last_active_date": "2026-05-01", "notice_period_days": 30,
        "endorsements_received": 43, "willing_to_relocate": True,
        "profile_completeness_score": 0.9
    }
}]

c1, c2 = st.columns([1, 3])
with c1:
    st.download_button(
        "⬇ Download sample format",
        data=json.dumps(_TEMPLATE, indent=2),
        file_name="candidate_template.json",
        mime="application/json",
        help="A ready-to-fill candidate in the expected JSON schema.",
    )
with c2:
    with st.expander("What format does the file need?"):
        st.markdown(
            "- A JSON **array** of candidates (or `.jsonl`, one object per line).\n"
            "- Each candidate needs `profile`, `career_history`, `skills`, and "
            "`redrob_signals` — the ranker scores **corroborated** evidence "
            "(skill duration, endorsements, real career history), so these nested "
            "fields matter.\n"
            "- CSV isn't supported: a flat row can't hold multiple jobs and "
            "per-skill evidence. Start from the template above and fill it in."
        )

uploaded = st.file_uploader("Candidate sample (.jsonl or .json)", type=["jsonl", "json"])
use_sample = st.checkbox("Use bundled sample_candidates.json", value=not uploaded)

# Which bundled set to run when no file is uploaded. The "showcase" is a
# curated 52-candidate set (real top engineers + planted stuffers + honeypots)
# so a reviewer can watch strong candidates rise AND the trap defenses fire in
# a single run — a far better demo than the neutral 50-candidate sample.
bundled_choice = "Standard sample (50 mixed candidates)"
if use_sample:
    bundled_choice = st.radio(
        "Which bundled set?",
        ["Standard sample (50 mixed candidates)",
         "Showcase (real top candidates + planted stuffers & honeypots)"],
        index=1,
        help="Showcase demonstrates ranking quality and trap defenses together.",
    )
top_n = st.slider("Shortlist size", 5, 100, 25)


def _read(file_or_path, is_path=False):
    raw = open(file_or_path, "r", encoding="utf-8").read() if is_path else file_or_path.getvalue().decode("utf-8")
    raw = raw.strip()
    if raw.startswith("["):                      # JSON array
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]   # JSONL


if st.button("Rank candidates", type="primary"):
    try:
        if uploaded is not None and not use_sample:
            cands = _read(uploaded)
        elif "Showcase" in bundled_choice:
            cands = _read("showcase_candidates.json", is_path=True)
        else:
            cands = _read("sample_candidates.json", is_path=True)
    except Exception as e:                        # noqa: BLE001
        st.error(f"Could not parse input: {e}")
        st.stop()

    if not isinstance(cands, list) or not cands:
        st.error("Expected a non-empty list of candidate objects "
                 "(a `.json` array, or `.jsonl` with one object per line).")
        st.stop()

    # Schema check: this ranker is built around the challenge candidate schema.
    # A file in a different shape (e.g. a generic resume-screening export) can't
    # be ranked meaningfully, so we say so clearly instead of returning noise.
    EXPECTED = {"profile", "career_history", "skills"}
    sample_keys = set(cands[0].keys()) if isinstance(cands[0], dict) else set()
    if not (EXPECTED & sample_keys):
        st.error(
            "This file doesn't match the expected candidate schema. "
            "Each record should include fields like `profile`, `career_history`, "
            "`skills`, and `redrob_signals` (see the bundled "
            "`sample_candidates.json`). The file you uploaded has a different "
            "shape, so it can't be ranked."
        )
        st.info("Tip: tick **Use bundled sample_candidates.json** above to see a "
                "working run, or upload a file in the challenge schema.")
        st.stop()

    n = min(len(cands), 100)
    cands = cands[:n]
    with st.spinner(f"Ranking {n} candidates…"):
        rows, _ = pipeline.rank(cands, top_n=min(top_n, n), verbose=False)

    st.success(f"Ranked {n} candidates → top {len(rows)}.")
    st.dataframe(rows, use_container_width=True)

    buf = io.StringIO()
    import csv
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    w.writerows(rows)
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="submission_sample.csv", mime="text/csv")
