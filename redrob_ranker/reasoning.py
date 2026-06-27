"""
reasoning.py — Generate a specific, honest, 1-2 sentence justification per pick.

Stage-4 review penalises: empty/identical/templated reasoning, hallucinated
skills, and reasoning that contradicts the rank. So every sentence here is built
ONLY from facts already in the candidate record, varies with the candidate's
actual profile, and surfaces genuine concerns for lower-ranked picks.
"""
from __future__ import annotations
from . import config as C


def _top_real_skills(c, k=3):
    """Highest-corroboration skills that are actually AI-relevant — never invented."""
    rel = []
    for s in c.get("skills", []):
        nm = s.get("name", "")
        low = nm.lower()
        if low in C.CORE_SKILLS or low in C.ML_SKILLS:
            dur = s.get("duration_months", 0) or 0
            end = s.get("endorsements", 0) or 0
            rel.append((dur + end, nm))
    rel.sort(reverse=True)
    return [nm for _, nm in rel[:k]]


def _product_employers(c):
    seen = []
    for h in c.get("career_history", []):
        comp = h.get("company", "")
        if comp.lower() in C.PRODUCT_COMPANIES and comp not in seen:
            seen.append(comp)
    return seen[:2]


def build_reasoning(c: dict, sc: dict) -> str:
    p = c.get("profile", {})
    title = p.get("current_title", "professional")
    yoe = p.get("years_of_experience", 0)
    beh = sc["behavioral"]

    bits = []
    # Lead clause: role + experience.
    bits.append(f"{title} with {yoe:.1f} yrs")

    # Corroborated AI skills (only real ones).
    skills = _top_real_skills(c)
    if skills:
        bits.append("strong in " + ", ".join(skills))

    # Product-company build evidence — the JD's ideal signal.
    employers = _product_employers(c)
    if employers and sc["career_info"]["build_evidence"] >= 0.34:
        bits.append("shipped ranking/retrieval work at " + ", ".join(employers))
    elif sc["career_info"]["build_evidence"] >= 0.5:
        bits.append("career shows hands-on retrieval/ranking build work")

    # Availability signal.
    if beh["days_idle"] <= 30 and beh["resp"] >= 0.6:
        bits.append(f"active recently, {beh['resp']:.0%} recruiter response")
    sentence = "; ".join(bits) + "."

    # Honest concern clause for weaker / lower picks.
    concerns = []
    if beh["resp"] < 0.3:
        concerns.append(f"low {beh['resp']:.0%} response rate")
    if beh["days_idle"] > 120:
        concerns.append(f"inactive ~{beh['days_idle']}d")
    if not beh["open"]:
        concerns.append("not flagged open-to-work")
    if beh["notice"] and beh["notice"] >= 90:
        concerns.append(f"{beh['notice']}d notice")
    if sc["components"]["location"] < 0.5:
        concerns.append("location/relocation gap")
    if sc["penalty_notes"]:
        concerns.append(sc["penalty_notes"][0])
    if concerns:
        sentence += " Concern: " + ", ".join(concerns[:2]) + "."

    # CSV-safety: collapse whitespace.
    return " ".join(sentence.split())
