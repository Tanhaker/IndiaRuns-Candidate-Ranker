# IndiaRuns — Senior AI Engineer Candidate Ranker (Agent Instructions)

**Team Fall Damage · India Runs — Data & AI Challenge (Redrob × Hack2Skill)**

This Space ranks candidates for a **Senior AI Engineer (Founding Team)** role the
way a recruiter would — by understanding *who actually fits*, not by counting
keywords. It runs **CPU-only, offline, with no LLM API calls**, and ranks the
bundled sample in a few seconds.

## What it does
Given a pool of candidate profiles, it returns a ranked shortlist with a
plain-English reason for every pick. Under the hood:
- **Hybrid retrieval** — BM25 + TF-IDF, fused by Reciprocal Rank Fusion (RRF).
- **Corroboration scoring** — a skill counts only if backed by real tenure,
  proficiency and endorsements, so keyword-stuffers collapse.
- **Trap defenses** — disqualifier penalties for stuffers, a behavioral
  availability multiplier, and a honeypot gate that sinks impossible profiles.
- **Weights tuned & validated** on a labeled eval set.

## How to use it
1. Open the Space and either upload a candidate file or use a bundled set.
2. Input format: a `.jsonl` file (one JSON candidate per line) or a `.json`
   array. Each candidate needs `profile`, `career_history`, `skills`, and
   `redrob_signals` fields. Click **Download sample format** in the app for a
   ready-to-fill template.
3. Two bundled demos are built in: a **Standard sample** (50 mixed candidates)
   and a **Showcase** (real top candidates plus planted stuffers and honeypots)
   that demonstrates ranking quality and trap defense in one run.
4. Choose a shortlist size and click **Rank candidates**. Download the ranked
   CSV (candidate_id, rank, score, reasoning).

## Notes
- CPU-only, no network, no LLM calls during ranking — mirrors the challenge's
  Stage-3 compute constraints.
- The score is normalized within whatever pool you run, so absolute values
  differ between a small sample and the full pool; the *ranking order* is what
  matters.

Repo: https://github.com/Tanhaker/IndiaRuns-Candidate-Ranker
