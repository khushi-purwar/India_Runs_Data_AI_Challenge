"""Honeypot / impossible-profile detection.

The dataset seeds ~80 candidates with subtly impossible profiles that are forced
to relevance tier 0 in the ground truth. Ranking any of them highly signals that
the system is matching tokens, not reading profiles. A top-100 honeypot rate
above 10% is an automatic disqualification, so we detect and hard-zero them.

We do NOT special-case known IDs - we detect impossibility from internal
inconsistencies that a careful human reviewer would catch.
"""
from __future__ import annotations

from datetime import date

CURRENT_YEAR = 2026


def _parse_year(value) -> int | None:
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


def honeypot_reasons(candidate: dict) -> list[str]:
    """Return a list of impossibility reasons; empty list means 'looks plausible'."""
    reasons: list[str] = []
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0) or 0)
    history = candidate.get("career_history", [])

    # 1. A single role lasting longer than the person's entire career.
    for job in history:
        dur_months = job.get("duration_months", 0) or 0
        if yoe > 0 and dur_months > (yoe * 12) + 18:
            reasons.append(
                f"role at {job.get('company','?')} lasts {dur_months} months but "
                f"total experience is only {yoe:.1f} years"
            )
            break

    # 2. Total tenure across roles wildly exceeds stated experience.
    total_months = sum((j.get("duration_months", 0) or 0) for j in history)
    # Allow generous overlap slack (concurrent roles) before flagging.
    if yoe > 0 and total_months > (yoe * 12) * 2.2 + 24:
        reasons.append(
            f"career tenure sums to {total_months} months vs {yoe:.1f} years stated"
        )

    # 3. Date inconsistencies within a role.
    for job in history:
        sd = job.get("start_date")
        ed = job.get("end_date")
        try:
            if sd:
                sdt = date.fromisoformat(sd)
                if sdt.year > CURRENT_YEAR:
                    reasons.append(f"role start date {sd} is in the future")
                if ed:
                    edt = date.fromisoformat(ed)
                    if edt < sdt:
                        reasons.append(f"role end date {ed} precedes start {sd}")
        except (ValueError, TypeError):
            pass

    # 4. A skill used for more months than the person has worked.
    for sk in candidate.get("skills", []):
        dur = sk.get("duration_months", 0) or 0
        if yoe > 0 and dur > (yoe * 12) + 24:
            reasons.append(
                f"skill '{sk.get('name','?')}' shows {dur} months use vs "
                f"{yoe:.1f} years total experience"
            )
            break

    # 5. "Expert" in many skills with zero endorsements AND zero/near-zero usage.
    suspicious_expert = 0
    for sk in candidate.get("skills", []):
        if (
            sk.get("proficiency") == "expert"
            and (sk.get("endorsements", 0) or 0) == 0
            and (sk.get("duration_months", 0) or 0) == 0
        ):
            suspicious_expert += 1
    if suspicious_expert >= 4:
        reasons.append(
            f"{suspicious_expert} 'expert' skills with 0 endorsements and 0 months used"
        )

    # 6. Education that finishes before it starts, or graduation implies the
    #    person started working before being born-plausible (entered workforce
    #    > ~10 years before finishing a degree).
    for edu in candidate.get("education", []):
        sy = _parse_year(edu.get("start_year"))
        ey = _parse_year(edu.get("end_year"))
        if sy and ey and ey < sy:
            reasons.append(f"education ends ({ey}) before it starts ({sy})")

    # 7. Earliest job start older than a plausible career given experience.
    starts = [j.get("start_date") for j in history if j.get("start_date")]
    if starts and yoe > 0:
        try:
            earliest = min(date.fromisoformat(s).year for s in starts)
            implied_years = CURRENT_YEAR - earliest
            # If they claim far more experience than their first job allows.
            if yoe > implied_years + 6:
                reasons.append(
                    f"claims {yoe:.1f} years but earliest role began only "
                    f"{implied_years} years ago"
                )
        except (ValueError, TypeError):
            pass

    return reasons


def is_honeypot(candidate: dict) -> bool:
    return len(honeypot_reasons(candidate)) > 0
