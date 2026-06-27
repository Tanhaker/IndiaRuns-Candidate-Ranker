"""
score.py — Combine extracted features into a single fit score per candidate.

Pipeline (per candidate):
    additive FIT  =  Sum_k  WEIGHTS[k] * component_k        (in [0, 1])
    final         =  FIT * behavioral_multiplier * honeypot_guard

The semantic component is computed in batch (TF-IDF over the whole pool) and
injected, because it needs the corpus; everything else is per-candidate.
"""
from __future__ import annotations
import math
from . import config as C
from . import features as F
from . import honeypot as H


def _squash(x, k=2.0):
    """Map an unbounded non-negative score into [0,1) with diminishing returns."""
    return 1.0 - math.exp(-x / k)


def score_candidate(c: dict, semantic: float) -> dict:
    title = F.title_strength(c)
    skf = F.skill_features(c)
    car = F.career_features(c)
    exp = F.experience_fit(c)
    edu = F.education_fit(c)
    loc = F.location_fit(c)
    beh_mult, beh = F.behavioral_multiplier(c)

    # --- skill_trust: corroborated core+ML skills, squashed ---------------
    skill_raw = skf["core"] * 1.0 + skf["ml"] * 0.5 + skf["bonus"] * 0.15
    skill_trust = _squash(skill_raw, k=6.0)

    # --- career: product-company build evidence ---------------------------
    career = (0.55 * car["build_evidence"]
              + 0.30 * car["product_frac"]
              + 0.15 * (1.0 - car["services_frac"]))

    # --- additive FIT -----------------------------------------------------
    fit = (C.WEIGHTS["title"] * title
           + C.WEIGHTS["skill_trust"] * skill_trust
           + C.WEIGHTS["semantic"] * semantic
           + C.WEIGHTS["career"] * career
           + C.WEIGHTS["experience"] * exp
           + C.WEIGHTS["education"] * edu
           + C.WEIGHTS["location"] * loc)

    # --- explicit JD anti-pattern penalties (multiplicative, gentle) ------
    penalty = 1.0
    notes = []
    # 1. Keyword stuffer: AI keywords present but neither title nor corroboration.
    if title <= C.TITLE_DEFAULT and skill_trust > 0.15 and semantic > 0.05:
        penalty *= 0.35
        notes.append("non-technical title with uncorroborated AI skills (stuffer pattern)")
    # 2. Framework enthusiast: heavy hype skills, thin real ML depth.
    if skf["hype"] >= 1.0 and skf["ml"] < 2.0:
        penalty *= 0.7
        notes.append("framework-hype skills without corroborated ML depth")
    # 3. Off-track specialism dominates with no IR/NLP corroboration.
    if skf["offtrack"] > skf["core"] + skf["ml"] and skf["core"] < 0.5:
        penalty *= 0.7
        notes.append("CV/speech focus without retrieval/NLP depth")
    # 4. Entire career at IT-services / consulting.
    if car["all_services"]:
        penalty *= 0.6
        notes.append("entire career at IT-services firms")
    # 5. Title-chaser / job-hopper.
    if car["job_hopper"]:
        penalty *= 0.85
        notes.append("short average tenure (title-chasing pattern)")

    # --- gentle out-of-band experience modifier ---------------------------
    # JD treats 5-9 yrs as a soft preference, not a gate, and will "seriously
    # consider" strong outliers. So we only lightly dock the extremes, leaving
    # genuinely elite seniors able to crack the top 100 but seated just below
    # comparable in-band engineers.
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0
    if yoe > 15:
        exp_mod = 0.88
    elif yoe > 13:
        exp_mod = 0.93
    elif yoe < 3:
        exp_mod = 0.90
    else:
        exp_mod = 1.0

    # --- honeypot guard ---------------------------------------------------
    # Two independent detectors, unioned: (a) the original profile heuristics
    # (zero-duration expert skills, coarse span mismatch), and (b) the strict
    # timeline-consistency checks (overlapping full-time roles, duration vs
    # dates, seniority floor, education ordering). Either firing sinks the
    # candidate; both are conservative so real profiles are not floored.
    hp = F.honeypot_flags(c, skf) + H.timeline_flags(c)
    guard = C.HONEYPOT["penalty"] if hp else 1.0

    final = fit * penalty * beh_mult * guard * exp_mod

    return {
        "candidate_id": c["candidate_id"],
        "final": final,
        "fit": fit,
        "components": {
            "title": title, "skill_trust": skill_trust, "semantic": semantic,
            "career": career, "experience": exp, "education": edu, "location": loc,
        },
        "skill_raw": skill_raw,
        "behavioral_mult": beh_mult,
        "behavioral": beh,
        "career_info": car,
        "penalty": penalty,
        "penalty_notes": notes,
        "honeypot": hp,
    }
