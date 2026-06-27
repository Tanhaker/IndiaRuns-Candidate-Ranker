"""
honeypot.py — Hard logical-consistency checks for subtly-impossible profiles.

Honeypots in this challenge are *designed to look attractive* (strong titles,
on-trend skills) but contain a subtle internal impossibility. A surface read —
or a scorer that only weighs how good the profile sounds — will rank them high
and risk the >10%-honeypots-in-top-100 instant disqualification.

These checks are deliberately CONSERVATIVE: each one fires only on a genuine
logical contradiction, never on a profile that is merely unusual. A real but
quirky candidate must never be flagged. Every check returns a short human-
readable reason string so the decision is auditable in the reasoning column.

Pure stdlib, no scoring decisions here — this module only *detects*; score.py
decides what penalty a detection earns.
"""
from __future__ import annotations
import datetime as _dt

# Dataset reference "now" — last_active dates in the pool run into 2026-05.
_TODAY = _dt.date(2026, 6, 1)

# A month of overlap below this is treated as noise (rounding, partial months),
# not an impossibility. Above it, two concurrent full-time roles is suspect.
_OVERLAP_TOLERANCE_M = 2

# Seniority floors: minimum plausible total experience (years) for a title.
# Below the floor with that title is a contradiction, not just fast promotion.
_SENIORITY_FLOORS = {
    "principal": 8,
    "staff": 7,
    "vp ": 9,
    "head of": 8,
    "director": 9,
    "distinguished": 10,
}


def _date(s):
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _months_between(a: _dt.date, b: _dt.date) -> int:
    """Whole months from a to b (b later). Negative if b precedes a."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def _job_intervals(career):
    """List of (start_date, end_date, role_dict) with parseable start dates."""
    out = []
    for h in career:
        s = _date(h.get("start_date"))
        if not s:
            continue
        e = _date(h.get("end_date")) or (_TODAY if h.get("is_current") else None)
        out.append((s, e, h))
    return out


def timeline_flags(c: dict) -> list[str]:
    """
    Return a list of hard-impossibility reasons. Empty list == passes all checks.
    Each reason is specific and fact-grounded so it can surface in reasoning.
    """
    flags = []
    prof = c.get("profile", {}) or {}
    yoe = prof.get("years_of_experience", 0) or 0
    career = c.get("career_history", []) or []
    intervals = _job_intervals(career)

    # --- 1. duration_months that contradicts its own start/end dates --------
    for s, e, h in intervals:
        if e is None:
            continue
        claimed = h.get("duration_months")
        if claimed is None:
            continue
        actual = _months_between(s, e)
        if actual < 0:
            flags.append("a role's end date precedes its start date")
            break
        # Allow generous slack (12m) for rounding / overlapping definitions;
        # only a gross mismatch is a contradiction.
        if claimed - actual > 18:
            flags.append(
                f"a role claims {claimed}mo but its dates span only {max(actual,0)}mo")
            break

    # --- 2. Impossible concurrent full-time roles ---------------------------
    # Two roles overlapping by more than tolerance, when neither is plainly an
    # advisory/part-time title, is not physically possible for one person.
    dated = sorted([(s, e, h) for (s, e, h) in intervals if e is not None],
                   key=lambda x: x[0])
    for i in range(len(dated) - 1):
        s1, e1, h1 = dated[i]
        s2, e2, h2 = dated[i + 1]
        overlap = _months_between(s2, e1)  # how far role-1 runs past role-2 start
        if overlap > _OVERLAP_TOLERANCE_M:
            t1 = (h1.get("title") or "").lower()
            t2 = (h2.get("title") or "").lower()
            parttime = ("advisor", "consultant", "freelance", "contract",
                        "mentor", "board", "intern")
            if not (any(p in t1 for p in parttime) or any(p in t2 for p in parttime)):
                flags.append(
                    f"two full-time roles overlap by {overlap}mo (one person)")
                break

    # --- 3. Career span far exceeding claimed experience --------------------
    starts = [s for (s, _, _) in intervals]
    if starts and yoe:
        span_yrs = (_TODAY - min(starts)).days / 365.25
        if span_yrs > yoe + 7:
            flags.append(
                f"first role began ~{span_yrs:.0f}y ago but only {yoe}y experience claimed")

    # --- 4. Sum of tenures wildly exceeding a single-track career -----------
    # If total months worked >> career span by a large margin (and roles are
    # not flagged part-time), the timeline cannot be reconciled.
    if starts:
        span_m = _months_between(min(starts), _TODAY)
        total_m = sum((h.get("duration_months") or 0) for h in career)
        if span_m > 0 and total_m - span_m > 36:
            flags.append(
                f"tenures sum to {total_m}mo across a {span_m}mo career window")

    # --- 5. Seniority incompatible with total experience --------------------
    cur_title = (prof.get("current_title") or "").lower()
    for key, floor in _SENIORITY_FLOORS.items():
        if key in cur_title and yoe and yoe + 1 < floor:
            flags.append(
                f"'{cur_title.strip()}' title with only {yoe}y total experience")
            break

    # --- 6. Education dated after the role that presupposes it ---------------
    # A degree completed *after* the candidate's first dated professional role
    # began is possible (part-time study); but a degree completed years after
    # the career START, paired with a senior current title, is a planted tell
    # only when it post-dates _every_ role. Keep this strict to avoid false hits.
    edu_end_years = []
    for ed in c.get("education", []) or []:
        ey = ed.get("end_year") or _date(ed.get("end_date")) and _date(ed.get("end_date")).year
        if isinstance(ey, int):
            edu_end_years.append(ey)
    if edu_end_years and starts:
        first_role_year = min(s.year for s in starts)
        latest_edu = max(edu_end_years)
        # Degree finished >3y after the LAST role would have ended is impossible
        # only if there are no current roles; we require it to post-date the
        # entire career to fire (very strict).
        last_end = max(
            [(e.year if e else _TODAY.year) for (_, e, _) in intervals] or [first_role_year])
        if latest_edu - last_end > 3 and not any(h.get("is_current") for h in career):
            flags.append(
                f"highest degree completed {latest_edu}, after the career ended {last_end}")

    return flags
