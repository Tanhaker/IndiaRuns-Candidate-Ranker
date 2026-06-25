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

uploaded = st.file_uploader("Candidate sample (.jsonl or .json)", type=["jsonl", "json"])
use_sample = st.checkbox("Use bundled sample_candidates.json", value=not uploaded)
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
        else:
            cands = _read("sample_candidates.json", is_path=True)
    except Exception as e:                        # noqa: BLE001
        st.error(f"Could not parse input: {e}")
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
