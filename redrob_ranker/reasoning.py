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


# Words that mark a concrete, role-relevant achievement in a description.
_ACH_MARKERS = (
    "pipeline", "ranking", "retrieval", "recommendation", "search", "rag",
    "embedding", "vector", "reranker", "ranker", "a/b", "fine-tun", "llm",
    "semantic", "recsys", "production", "shipped", "latency", "ndcg", "recall",
)


def _achievement(c):
    """
    Pull ONE concrete, real achievement clause from the candidate's own career
    descriptions — never invented. Prefers the current role, prefers a sentence
    that names a system AND carries a number (scale/metric), and trims it to a
    short, CSV-safe clause. Returns "" if nothing solid is present.
    """
    history = c.get("career_history", []) or []
    ordered = [h for h in history if h.get("is_current")] + \
              [h for h in history if not h.get("is_current")]

    best = ""
    best_score = 0
    for h in ordered:
        desc = (h.get("description") or "").strip()
        if not desc:
            continue
        # first 1-2 sentences only; earlier sentences are usually the clean
        # headline achievement, so give position 0 a bonus to prefer it.
        sentences = desc.replace("\n", " ").split(". ")[:2]
        for pos, sentence in enumerate(sentences):
            s = sentence.strip().rstrip(".")
            low = s.lower()
            if len(s) < 20:
                continue
            # skip fragments that lean on a prior sentence ("The architecture…")
            if low.startswith(("the architecture", "this ", "it ", "these ")):
                continue
            marker_hits = sum(1 for m in _ACH_MARKERS if m in low)
            has_number = any(ch.isdigit() for ch in s)
            score = marker_hits + (2 if has_number else 0) + (1 if pos == 0 else 0)
            if score > best_score:
                best_score, best = score, s
        if best_score >= 3 and ordered.index(h) == 0:
            break  # a strong current-role achievement is enough

    if best_score < 1:
        return ""
    # trim to a concise clause at a word boundary (~95 chars)
    if len(best) > 95:
        best = best[:95].rsplit(" ", 1)[0] + "…"
    # lowercase a leading capital so it reads inside the sentence
    if best and best[0].isupper() and not best[:3].isupper():
        best = best[0].lower() + best[1:]
    return best


def build_reasoning(c: dict, sc: dict) -> str:
    p = c.get("profile", {})
    title = p.get("current_title", "professional")
    yoe = p.get("years_of_experience", 0)
    beh = sc["behavioral"]

    # --- Honeypot / impossible-profile: the flag IS the explanation. --------
    # When the guard fires, a positive description would contradict the rank.
    # Lead with the impossibility so the reasoning matches the decision: the
    # profile may read as strong, but it was sunk because it cannot be real.
    flags = sc.get("honeypot") or []
    if flags:
        reason = flags[0]
        sentence = (
            f"FLAGGED as inconsistent — {reason}. "
            f"Profile presents as {title} ({yoe:.1f} yrs) but the record is "
            f"internally impossible, so it is sunk regardless of stated skills."
        )
        return " ".join(sentence.split())

    bits = []
    # Lead clause: role + experience.
    bits.append(f"{title} with {yoe:.1f} yrs")

    # Corroborated AI skills (only real ones).
    skills = _top_real_skills(c)
    if skills:
        bits.append("strong in " + ", ".join(skills))

    # Product-company build evidence — now grounded in a REAL achievement
    # pulled from the candidate's own career history, not a generic phrase.
    employers = _product_employers(c)
    achievement = _achievement(c)
    if achievement and employers:
        bits.append(f"at {employers[0]}, {achievement}")
    elif achievement:
        bits.append(achievement)
    elif employers and sc["career_info"]["build_evidence"] >= 0.34:
        bits.append("shipped ranking/retrieval work at " + ", ".join(employers))
    elif sc["career_info"]["build_evidence"] >= 0.5:
        bits.append("career shows hands-on retrieval/ranking build work")

    # Availability signal.
    if beh["days_idle"] <= 30 and beh["resp"] >= 0.6:
        bits.append(f"active recently, {beh['resp']:.0%} recruiter response")

    # Independent verification — cite Redrob's own platform evidence when it
    # exists (assessment score on role-relevant skills, GitHub activity). This
    # is corroboration the candidate can't self-author, so it strengthens the
    # "why" without affecting the score.
    v = sc.get("verification") or {}
    if v.get("assess_relevant") is not None:
        bits.append(f"independently assessed {v['assess_relevant']:.0f}/100 on relevant skills")
    if v.get("github") is not None and v["github"] >= 40:
        bits.append(f"active GitHub ({v['github']:.0f}/100)")

    sentence = "; ".join(bits) + "."

    # Honest concern clause for weaker / lower picks.
    concerns = []
    if v.get("note"):
        concerns.append(v["note"])
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
