# Redrob — Intelligent Candidate Discovery & Ranking

Ranks the top-100 best-fit candidates from a 100,000-profile pool for the
**Senior AI Engineer — Founding Team** job description, the way a great recruiter
would: by understanding the *role behind the words*, not by counting keywords.

```bash
# 1. install (CPU-only, two small deps)
pip install -r requirements.txt

# 2. reproduce the submission (≈40s on a laptop CPU; well under the 5-min budget)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. (optional) validate against the official spec
python validate_submission.py submission.csv
```

`candidates.jsonl` (or `candidates.jsonl.gz`) is the released pool. The command
above is the single, hidden-step-free reproduction command used at Stage-3.

---

## TL;DR of the approach

A **transparent hybrid ranker**: an interpretable rule-based scorer (the
backbone) fused with a **TF-IDF semantic retrieval lane**, then modulated by a
**behavioral-availability multiplier** and protected by **honeypot and
keyword-stuffer guards**. Everything is CPU-only and runs offline — no LLM calls,
no GPU — exactly the latency/quality posture the JD says it wants in production.

```
final = FIT_score × anti_pattern_penalty × behavioral_multiplier × honeypot_guard × experience_mod

FIT_score = 0.26·title  + 0.22·skill_trust + 0.20·semantic + 0.16·career
          + 0.08·experience + 0.04·education + 0.04·location
```

Result on the public pool: **0 honeypots** and **0 non-technical "stuffer"
titles** in the top-100; the planted keyword-stuffer (a "Project Manager"
stuffed with AI skills) sinks to rank **87,613 / 100,000**; reasonings are
**100/100 unique** and fact-grounded.

---

## Why this design (reading the JD the way it asks to be read)

The JD is unusually explicit that the "right answer" is **not** "most AI
keywords." It plants four trap families and tells you they exist. Each one maps
to a specific defence in the scorer:

| What the JD/data does | The trap | Our defence |
|---|---|---|
| Marketing/HR/etc. profiles stuffed with AI skills | **Keyword stuffer** | `title` tier (role must be on-track) + `skill_trust` (skills only count when corroborated) + explicit stuffer penalty |
| Strong builders who *don't* use buzzwords | **Plain-language Tier-5** | `semantic` lane + `career` build-evidence read descriptions, not just skill tags |
| Identical profiles differing only on engagement | **Behavioral twins** | `behavioral_multiplier` from the 23 Redrob signals |
| Subtly impossible profiles | **~80 honeypots** | `honeypot_guard` impossibility checks (and corroboration starves them anyway) |

### The decisive idea: corroboration, not presence

A skill contributes in proportion to *evidence that it is real*:

```
skill_trust(skill) = proficiency_weight × √(duration_months) × (1 + log(1+endorsements)/3)
```

The planted stuffer lists `Recommendation Systems`, `Pinecone`, `Vector Search`
— but at `beginner/intermediate` proficiency, `0–16` months used, `3–4`
endorsements. A real target lists the same skills at `expert`, `50–88` months,
`40+` endorsements. The √duration term means *listing a skill you've barely used
earns almost nothing*. This single mechanism does most of the anti-stuffing work;
the title tier and explicit penalty are belt-and-braces.

### Behavioral signals as a *multiplier*, per the signals doc

> "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
> response rate is, for hiring purposes, not actually available."

We fold `recruiter_response_rate`, last-active recency, `open_to_work`,
profile completeness, recruiter saves, interview completion and notice period
into a single multiplier in `[0.55, 1.08]` — it can sink an unavailable
star but only gently lift an available one.

### Honeypots

Forced to relevance tier-0 in the hidden ground truth; >10% in the top-100 is an
instant disqualification. Two independent detectors run, and **either** firing
sinks a candidate (score ×`0.02` — sunk, not hard-deleted, so a near-miss real
candidate is never destroyed):

1. **Profile heuristics** (`features.honeypot_flags`) — the documented
   signatures: ≥3 "expert" skills with 0 months used; a career timeline
   exceeding claimed years of experience; a single role longer than the whole
   career.
2. **Strict timeline-consistency checks** (`honeypot.timeline_flags`) — hard
   logical contradictions a "looks-good" read misses: a role whose
   `duration_months` contradicts its own start/end dates; two full-time roles
   overlapping for one person; tenures summing past the career window; a
   Staff/Principal/Director title with too few total years; a degree dated after
   the entire career ended. Every check is conservative and returns a specific,
   auditable reason.

On the full 100K pool the two agree on 19 profiles, the timeline checker catches
1 more the heuristics miss, and it supplies sharper impossibility reasons for the
shared ones. Corroboration already starves honeypots of score; this is layered
insurance against the one disqualifying failure mode.

---

## Measuring changes: the eval harness (`eval/`)

There is no live leaderboard and only 3 submissions, so every change is a guess
until measured. The `eval/` harness reproduces the challenge's exact blend
(`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`) against a validation set
you label by hand:

```bash
python -m eval.make_labels      # writes eval/labels.csv (then fill your_tier 0-4)
python -m eval.evaluate         # prints the composite for the current ranker
```

Use it as a private leaderboard: change a weight in `config.py`, re-run, keep the
change only if the number rises. See `eval/README.md` for the tier scale and
caveats (the public sample is stuffer-heavy, so compare *versions*, not absolutes).

---

## Architecture / repo layout

```
rank.py                     # single-command CLI entrypoint (Stage-3 reproduce command)
redrob_ranker/
  config.py                 # ALL domain knowledge: weights, vocabularies, company lists, JD query
  features.py               # pure feature extraction from a candidate dict (stdlib only)
  score.py                  # combines features → final score (rules + penalties + guards)
  honeypot.py               # strict timeline-consistency checks for impossible profiles
  semantic — in pipeline.py # TF-IDF JD-match lane over the whole pool
  reasoning.py              # specific, non-templated, fact-grounded justification per pick
  pipeline.py               # load → semantic → score → rank → write CSV
eval/                       # local eval harness (NDCG@k/MAP/P@k vs hand labels)
  make_labels.py            # build a labeling worksheet from the public sample
  evaluate.py               # score the live ranker (or a CSV) with the official blend
  metrics.py                # NDCG@k, MAP, P@k, composite — pure stdlib
app.py                      # Streamlit sandbox (≤100-candidate sample → ranked CSV)
Dockerfile                  # reproducible build for the sandbox / Stage-3 container
tests/test_smoke.py         # sanity tests (stuffer sinks, target floats, CSV valid)
submission_metadata.yaml    # portal metadata mirror
requirements.txt
```

### Compute posture (matches submission_spec Section 3)
- **CPU only, no GPU.** Pure NumPy + scikit-learn TF-IDF.
- **No network at runtime.** TF-IDF is fit on the pool itself; nothing is
  downloaded. (No hosted-LLM calls anywhere in the ranking path.)
- **~40 s wall-clock** for 100K candidates; **<1 GB** RAM. Comfortably inside
  the 5-min / 16 GB budget.

### Why TF-IDF and not dense embeddings?
The ranking step must run offline with no model download inside the sandbox.
TF-IDF gives a strong, fully-reproducible semantic lane with zero artifacts to
ship. The architecture is **embedding-swappable**: `pipeline.semantic_scores()`
is the only seam — drop in pre-computed BGE/E5 vectors (documented as offline
pre-computation) and the rest of the system is unchanged. That mirrors the
hybrid BM25-plus-dense retrieval the JD describes, and is the obvious production
upgrade path.

---

## Reasoning column (Stage-4)

Each justification is assembled **only** from facts in the candidate record:
role + years, top *corroborated* AI skills, product-company build evidence, and
real engagement numbers — with an honest `Concern:` clause for genuine gaps
(low response rate, long inactivity, 90-day notice, location/relocation gap).
No skill is ever mentioned that isn't in the profile, and the tone tracks the
rank. All 100 are distinct.

---

## Reproducing / testing locally

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv      # → "Submission is valid."
pytest -q                                          # smoke tests
python -m eval.make_labels && python -m eval.evaluate   # local quality score
```

## Sandbox

`app.py` is a Streamlit app that accepts a small candidate sample (≤100, upload
or the bundled `sample_candidates.json`) and returns the ranked CSV in seconds —
the lightweight reproducibility check described in submission_spec Section 10.5.
Run locally with `streamlit run app.py`, or deploy free to HuggingFace Spaces /
Streamlit Cloud. A `Dockerfile` is included for a `docker run` recipe.

## AI tools

Architecture discussion, code review, and drafting were done with AI assistance
(declared in `submission_metadata.yaml`). All design decisions, weight
calibration against the trap profiles, and the engineering are the team's own.
No candidate data was sent to any hosted LLM, and **no LLM is called anywhere in
the ranking path** — a hard requirement of the compute budget.
