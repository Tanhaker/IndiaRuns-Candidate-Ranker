"""
features.py — Turn a raw candidate dict into a flat, scoreable feature record.

Pure functions, no scoring decisions here — just extraction and light parsing.
Kept dependency-free (stdlib only) so it is trivial to reason about and reproduce.
"""
from __future__ import annotations
import datetime as _dt
from . import config as C

_TODAY = _dt.date(2026, 6, 1)  # dataset reference "now" (last_active dates run to 2026-05)


def _year(date_str):
    try:
        return int(str(date_str)[:4])
    except (TypeError, ValueError):
        return None


def _parse_date(date_str):
    try:
        return _dt.date.fromisoformat(str(date_str)[:10])
    except (TypeError, ValueError):
        return None


def candidate_text(c: dict) -> str:
    """Full searchable text blob for the semantic (TF-IDF) lane."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for h in c.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
    parts.extend(s.get("name", "") for s in c.get("skills", []))
    parts.extend(ce.get("name", "") for ce in c.get("certifications", []) or [])
    return " ".join(parts).lower()


def title_strength(c: dict) -> float:
    """Best title-tier match across current + historical titles."""
    titles = [c.get("profile", {}).get("current_title", "")]
    titles += [h.get("title", "") for h in c.get("career_history", [])]
    titles = [t.lower() for t in titles if t]
    best = C.TITLE_DEFAULT
    # Current title counts double-strength vs historical (recency of role matters).
    cur = titles[0] if titles else ""
    for score, keys in C.TITLE_TIERS.items():
        for k in keys:
            if k in cur:
                best = max(best, score)
            elif any(k in t for t in titles[1:]):
                best = max(best, score * 0.85)  # historical-only, slight discount
    return best


def skill_features(c: dict) -> dict:
    """
    Corroborated skill signal. A skill only 'counts' in proportion to evidence
    that it is real: proficiency x sqrt(duration_months) x (1 + log endorsements).
    This is the core defence against keyword stuffing — a skill listed with
    0 months used and 0 endorsements contributes ~nothing.
    """
    import math
    prof_w = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.85, "expert": 1.0}
    core = ml = bonus = 0.0
    hype = offtrack = 0.0
    expert_zero_dur = 0
    for s in c.get("skills", []):
        name = s.get("name", "").lower()
        dur = s.get("duration_months", 0) or 0
        end = s.get("endorsements", 0) or 0
        pw = prof_w.get(s.get("proficiency", "beginner"), 0.25)
        if s.get("proficiency") == "expert" and dur == 0:
            expert_zero_dur += 1
        # Evidence-weighted strength of this single skill.
        trust = pw * math.sqrt(min(dur, 72)) * (1.0 + math.log1p(end) / 3.0)
        if name in C.CORE_SKILLS:
            core += trust
        elif name in C.ML_SKILLS:
            ml += trust
        elif name in C.BONUS_SKILLS:
            bonus += trust
        if name in C.FRAMEWORK_HYPE_SKILLS:
            hype += pw
        if name in C.OFFTRACK_SKILLS:
            offtrack += pw
    return {
        "core": core, "ml": ml, "bonus": bonus,
        "hype": hype, "offtrack": offtrack,
        "expert_zero_dur": expert_zero_dur,
    }


def career_features(c: dict) -> dict:
    """Product-vs-services mix, build-evidence, tenure pattern."""
    hist = c.get("career_history", [])
    n = max(len(hist), 1)
    services = product = 0
    build_hits = 0
    durations = []
    BUILD_TERMS = ("ranking", "retrieval", "recommendation", "search", "embedding",
                   "vector", "relevance", "learning-to-rank", "learning to rank",
                   "a/b test", "ndcg", "shipped", "production", "deployed")
    for h in hist:
        comp = h.get("company", "").lower()
        desc = h.get("description", "").lower()
        if comp in C.SERVICES_COMPANIES:
            services += 1
        elif comp in C.PRODUCT_COMPANIES:
            product += 1
        if any(t in desc for t in BUILD_TERMS):
            build_hits += 1
        durations.append(h.get("duration_months", 0) or 0)
    avg_tenure = sum(durations) / n if durations else 0
    return {
        "services_frac": services / n,
        "product_frac": product / n,
        "build_evidence": build_hits / n,      # fraction of roles describing real build work
        "all_services": services == n and n >= 2,
        "avg_tenure_m": avg_tenure,
        "n_roles": len(hist),
        "job_hopper": (avg_tenure < 18 and len(hist) >= 4),
    }


def experience_fit(c: dict) -> float:
    """Triangular preference peaking at 6-8 yrs, acceptable 5-9, soft outside."""
    y = c.get("profile", {}).get("years_of_experience", 0) or 0
    if 6 <= y <= 8:
        return 1.0
    if 5 <= y < 6 or 8 < y <= 9:
        return 0.9
    if 4 <= y < 5 or 9 < y <= 11:
        return 0.7
    if 3 <= y < 4 or 11 < y <= 13:
        return 0.45
    return 0.25


def education_fit(c: dict) -> float:
    tiers = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45, "unknown": 0.5}
    best = 0.5
    for e in c.get("education", []):
        best = max(best, tiers.get(e.get("tier", "unknown"), 0.5))
    return best


def location_fit(c: dict) -> float:
    p = c.get("profile", {})
    loc = (p.get("location", "") + " " + p.get("country", "")).lower()
    relocate = c.get("redrob_signals", {}).get("willing_to_relocate", False)
    in_india = "india" in loc
    in_metro = any(city in loc for city in C.TARGET_CITIES)
    if in_metro:
        return 1.0
    if in_india and relocate:
        return 0.9
    if in_india:
        return 0.75
    if relocate:
        return 0.45            # abroad but willing — JD says case-by-case
    return 0.2                 # abroad, not willing, no visa sponsorship


def behavioral_multiplier(c: dict) -> tuple[float, dict]:
    """
    Availability/engagement modifier in [floor, ceil]. Combines responsiveness,
    recency, intent, and recruiter pull. Returns (multiplier, sub-signals) so the
    reasoning layer can cite the actual numbers.
    """
    s = c.get("redrob_signals", {})
    resp = s.get("recruiter_response_rate", 0) or 0
    last = _parse_date(s.get("last_active_date"))
    days_idle = (_TODAY - last).days if last else 365
    recency = max(0.0, 1.0 - days_idle / 180.0)         # 0 by ~6 months idle
    open_flag = 1.0 if s.get("open_to_work_flag") else 0.0
    completeness = (s.get("profile_completeness_score", 0) or 0) / 100.0
    saved = min((s.get("saved_by_recruiters_30d", 0) or 0) / 20.0, 1.0)
    interview = s.get("interview_completion_rate", 0) or 0
    notice = s.get("notice_period_days", 90) or 90
    notice_fit = 1.0 if notice <= 30 else (0.7 if notice <= 60 else 0.4)

    raw = (0.34 * resp + 0.20 * recency + 0.14 * open_flag + 0.10 * completeness
           + 0.08 * saved + 0.08 * interview + 0.06 * notice_fit)
    lo, hi = C.BEHAVIORAL["floor"], C.BEHAVIORAL["ceil"]
    mult = lo + (hi - lo) * raw
    return mult, {
        "resp": resp, "days_idle": days_idle, "open": bool(s.get("open_to_work_flag")),
        "notice": notice, "saved": s.get("saved_by_recruiters_30d", 0),
        "github": s.get("github_activity_score", -1),
    }


def honeypot_flags(c: dict, skf: dict) -> list[str]:
    """Detect subtly-impossible profiles (forced to tier 0 in ground truth)."""
    flags = []
    y = c.get("profile", {}).get("years_of_experience", 0) or 0
    if skf["expert_zero_dur"] >= C.HONEYPOT["expert_zero_dur_min"]:
        flags.append(f"{skf['expert_zero_dur']} 'expert' skills with 0 months used")
    starts = [_year(h.get("start_date")) for h in c.get("career_history", [])]
    starts = [s for s in starts if s]
    if starts:
        span = 2026 - min(starts)
        if span > y + C.HONEYPOT["career_span_slack_yrs"]:
            flags.append(f"career spans {span}y but only {y}y experience claimed")
    for h in c.get("career_history", []):
        if (h.get("duration_months", 0) or 0) > y * 12 + C.HONEYPOT["role_over_yoe_slack_m"]:
            flags.append("single role longer than total career")
            break
    return flags


def verification_signal(c: dict, skf: dict) -> tuple[float, dict]:
    """
    Independent corroboration from Redrob's OWN platform signals — the strongest
    kind of evidence, because the candidate cannot self-author it.

    Two sources, used only when present (absent => neutral, never penalised):
      * skill_assessment_scores — Redrob's independent 0-100 test per skill.
        High scores on role-relevant skills CORROBORATE the profile; a high
        claimed proficiency paired with a LOW assessment EXPOSES a stuffer.
      * github_activity_score — objective evidence of real building (0-100;
        -1/None = no GitHub linked = neutral).

    Returns (modifier, info) where modifier is a gentle multiplier centred on
    1.0: corroboration nudges up, contradiction nudges down, missing = 1.0.
    """
    s = c.get("redrob_signals", {}) or {}
    info = {"assess_relevant": None, "github": None, "note": ""}

    # --- skill assessments on role-relevant skills -----------------------
    assess = s.get("skill_assessment_scores") or {}
    relevant = []
    for name, val in assess.items():
        low = (name or "").lower()
        if low in C.CORE_SKILLS or low in C.ML_SKILLS:
            try:
                relevant.append(float(val))
            except (TypeError, ValueError):
                pass
    assess_mod = 0.0
    if relevant:
        avg = sum(relevant) / len(relevant)
        info["assess_relevant"] = round(avg, 1)
        # map 0-100 assessment to a [-0.10, +0.10] nudge around 50.
        assess_mod = max(-0.10, min(0.10, (avg - 50.0) / 500.0))
        # contradiction check: profile claims strong skills but Redrob's own
        # test says otherwise -> firmer downward nudge (independent exposure).
        if avg < 30 and skf.get("core", 0) + skf.get("ml", 0) >= 2.0:
            assess_mod = -0.12
            info["note"] = "claims strong skills but low independent assessment"

    # --- github activity --------------------------------------------------
    gh = s.get("github_activity_score", -1)
    gh_mod = 0.0
    if gh is not None and gh >= 0:
        info["github"] = gh
        gh_mod = min(0.06, (gh / 100.0) * 0.06)   # up to +0.06 for prolific GH

    modifier = 1.0 + assess_mod + gh_mod
    return max(0.85, min(1.12, modifier)), info
