"""Per-candidate scoring.

Each candidate gets several interpretable component scores in [0, 1], a weighted
base-fit score, and two multipliers:

* ``behavioral_multiplier`` - availability/engagement (an unreachable candidate
  is not actually hireable, however good on paper).
* ``disqualifier_multiplier`` - the JD's explicit anti-signals (consulting-only
  career, CV/speech-only, title-chasing, academia-only).

Honeypots are hard-zeroed by the caller. Every number used for the final score is
also stored in a ``detail`` dict so the reasoning layer can cite real facts.
"""
from __future__ import annotations

import math
from datetime import date

from . import jd
from .honeypot import honeypot_reasons
from .text import career_prose, skills_text

CURRENT = date(2026, 6, 20)


# --------------------------------------------------------------------------- #
# Component scorers
# --------------------------------------------------------------------------- #
def score_title(candidate: dict) -> tuple[float, dict]:
    """Decisive role-fit signal from current + historical titles."""
    profile = candidate.get("profile", {})
    cur = (profile.get("current_title", "") or "").lower()

    # Most negative match on the current title dominates - this is what stops a
    # "Marketing Manager" with a stuffed AI skill list from ranking.
    neg = min(
        (v for k, v in jd.NEGATIVE_TITLES.items() if k in cur),
        default=None,
    )
    pos = max(
        (v for k, v in jd.POSITIVE_TITLES.items() if k in cur),
        default=0.0,
    )

    if neg is not None:
        current_score = neg
    elif pos > 0:
        current_score = pos
    else:
        current_score = 0.35  # unknown/ambiguous title

    # Historical exposure to engineering titles gives partial credit (career
    # pivots), but never overrides a clearly non-technical current role.
    hist_pos = 0.0
    for job in candidate.get("career_history", []):
        t = (job.get("title", "") or "").lower()
        hist_pos = max(hist_pos, max(
            (v for k, v in jd.POSITIVE_TITLES.items() if k in t), default=0.0))

    score = 0.8 * current_score + 0.2 * hist_pos
    if neg is not None:
        score = min(score, current_score + 0.10)  # cap pivots from non-tech roles
    return max(0.0, min(1.0, score)), {"current_title": profile.get("current_title", "")}


def _weighted_term_hits(text: str, terms: dict[str, float]) -> float:
    text = text.lower()
    return sum(w for term, w in terms.items() if term in text)


def score_domain(candidate: dict, semantic_sim: float) -> tuple[float, dict]:
    """Evidence of retrieval/ranking/recsys/NLP work + semantic match.

    Prose (summary + role descriptions) is weighted ~3x the skills array because
    real narratives are far harder to fake than a comma-separated list.
    """
    prose = career_prose(candidate)
    skills = skills_text(candidate)

    prose_hits = _weighted_term_hits(prose, jd.CORE_DOMAIN_TERMS)
    skill_hits = _weighted_term_hits(skills, jd.CORE_DOMAIN_TERMS)
    bonus_hits = _weighted_term_hits(prose + " " + skills, jd.BONUS_TERMS)

    # Saturating transforms so a handful of strong, genuine mentions count, but a
    # keyword dump does not run away with the score.
    prose_score = 1.0 - math.exp(-prose_hits / 2.5)
    skill_score = 1.0 - math.exp(-skill_hits / 3.5)
    bonus_score = 1.0 - math.exp(-bonus_hits / 3.0)

    evidence = 0.62 * prose_score + 0.20 * skill_score + 0.18 * bonus_score
    # Blend hand-coded evidence with the semantic match (captures plain-language
    # Tier-5 candidates who describe the work without the vocabulary).
    score = 0.65 * evidence + 0.35 * semantic_sim
    return max(0.0, min(1.0, score)), {
        "prose_hits": round(prose_hits, 1),
        "semantic_sim": round(semantic_sim, 3),
    }


def score_experience(candidate: dict) -> tuple[float, dict]:
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0)
    if jd.EXP_IDEAL_LOW <= yoe <= jd.EXP_IDEAL_HIGH:
        s = 1.0
    elif yoe < jd.EXP_IDEAL_LOW:
        # Ramp from soft-low up to ideal-low.
        s = max(0.0, (yoe - jd.EXP_SOFT_LOW) / (jd.EXP_IDEAL_LOW - jd.EXP_SOFT_LOW))
        s = 0.25 + 0.75 * max(0.0, min(1.0, s))
    else:
        # Above ideal: gentle decay (seniority is fine, just not ideal band).
        over = yoe - jd.EXP_IDEAL_HIGH
        s = max(0.30, 1.0 - 0.10 * over)
    return max(0.0, min(1.0, s)), {"yoe": round(yoe, 1)}


def score_skills(candidate: dict) -> tuple[float, dict]:
    """Relevant-skill coverage with a trust weight against keyword stuffing.

    A skill counts more when it is backed by endorsements, real duration, and a
    Redrob assessment score. "Expert" claims with no endorsements and no usage
    count for almost nothing.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}

    total_trust = 0.0
    relevant_count = 0
    for sk in candidate.get("skills", []):
        name = (sk.get("name", "") or "").lower()
        if name not in jd.RELEVANT_SKILLS:
            continue
        relevant_count += 1
        endorsements = sk.get("endorsements", 0) or 0
        duration = sk.get("duration_months", 0) or 0
        prof = sk.get("proficiency", "beginner")
        prof_w = {"beginner": 0.4, "intermediate": 0.7,
                  "advanced": 0.9, "expert": 1.0}.get(prof, 0.5)

        # Trust = does anything corroborate the claim?
        trust = 0.15
        trust += min(0.35, endorsements / 40.0)
        trust += min(0.30, duration / 36.0)
        assess = assessments.get(sk.get("name", ""), None)
        if assess is not None:
            trust += 0.20 * (assess / 100.0)
        total_trust += prof_w * min(1.0, trust)

    # Saturating coverage.
    score = 1.0 - math.exp(-total_trust / 2.2)
    return max(0.0, min(1.0, score)), {"relevant_skills": relevant_count}


def score_product(candidate: dict) -> tuple[float, dict]:
    """Product-company vs services/consulting career signal."""
    companies = [(j.get("company", "") or "").lower()
                 for j in candidate.get("career_history", [])]
    if not companies:
        return 0.5, {"consulting_ratio": 0.0}
    consulting = sum(
        1 for c in companies if any(f in c for f in jd.CONSULTING_FIRMS)
    )
    ratio = consulting / len(companies)
    # Entirely consulting -> low; some product experience -> high.
    score = 1.0 - 0.85 * ratio
    return max(0.0, min(1.0, score)), {"consulting_ratio": round(ratio, 2)}


def score_location(candidate: dict) -> tuple[float, dict]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    loc = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    in_metro = any(city in loc for city in jd.PREFERRED_LOCATIONS)
    willing = bool(signals.get("willing_to_relocate", False))

    if in_metro:
        s = 1.0
    elif "india" in country and willing:
        s = 0.85
    elif "india" in country:
        s = 0.6
    elif willing:
        s = 0.45  # outside India but mobile; JD is case-by-case, no visa sponsor
    else:
        s = 0.25
    return s, {"location": profile.get("location", ""), "willing_to_relocate": willing}


def score_education(candidate: dict) -> tuple[float, dict]:
    best = 0.4
    field_bonus = 0.0
    for edu in candidate.get("education", []):
        tier = edu.get("tier", "unknown")
        best = max(best, {
            "tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6,
            "tier_4": 0.45, "unknown": 0.4,
        }.get(tier, 0.4))
        field = (edu.get("field_of_study", "") or "").lower()
        if any(k in field for k in (
                "computer", "data", "machine learning", "artificial",
                "statistic", "mathematic", "electronics", "information")):
            field_bonus = 0.1
    return max(0.0, min(1.0, best * 0.85 + field_bonus)), {}


# --------------------------------------------------------------------------- #
# Multipliers
# --------------------------------------------------------------------------- #
def behavioral_multiplier(candidate: dict) -> tuple[float, dict]:
    """Availability/engagement modifier in roughly [0.45, 1.08]."""
    s = candidate.get("redrob_signals", {}) or {}

    # Recency of activity.
    recency = 0.5
    last = s.get("last_active_date")
    if last:
        try:
            days = (CURRENT - date.fromisoformat(last)).days
            if days <= 30:
                recency = 1.0
            elif days <= 90:
                recency = 0.85
            elif days <= 180:
                recency = 0.6
            else:
                recency = 0.3
        except (ValueError, TypeError):
            pass

    resp = float(s.get("recruiter_response_rate", 0) or 0)
    resp_score = min(1.0, 0.2 + resp)  # 0 -> 0.2, 0.8 -> 1.0

    open_flag = 1.0 if s.get("open_to_work_flag") else 0.7
    interview = float(s.get("interview_completion_rate", 0.5) or 0.5)
    completeness = float(s.get("profile_completeness_score", 50) or 50) / 100.0
    verified = 1.0 if (s.get("verified_email") and s.get("verified_phone")) else 0.92

    core = (
        0.34 * recency
        + 0.30 * resp_score
        + 0.14 * interview
        + 0.12 * completeness
        + 0.10 * open_flag
    )
    mult = 0.5 + 0.55 * core
    mult *= verified
    mult = max(0.45, min(1.08, mult))
    return mult, {
        "recency_days": (CURRENT - date.fromisoformat(last)).days if last else None,
        "response_rate": round(resp, 2),
        "open_to_work": bool(s.get("open_to_work_flag")),
    }


def disqualifier_multiplier(candidate: dict, domain_detail: dict) -> tuple[float, list[str]]:
    """Apply the JD's explicit anti-signals as a multiplicative penalty."""
    mult = 1.0
    flags: list[str] = []
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])

    # Consulting-only career.
    companies = [(j.get("company", "") or "").lower() for j in history]
    if companies and all(any(f in c for f in jd.CONSULTING_FIRMS) for c in companies):
        mult *= 0.45
        flags.append("entire career at consulting/services firms")

    # CV / speech / robotics specialist without NLP/IR evidence.
    skill_names = {(s.get("name", "") or "").lower() for s in candidate.get("skills", [])}
    cv_skills = skill_names & jd.CV_SPEECH_ROBOTICS_SKILLS
    nlp_evidence = domain_detail.get("prose_hits", 0) >= 1.5 or {
        "nlp", "information retrieval", "semantic search", "rag",
        "fine-tuning llms", "embeddings",
    } & skill_names
    if len(cv_skills) >= 3 and not nlp_evidence:
        mult *= 0.6
        flags.append("primarily computer-vision/speech without NLP/IR exposure")

    # Title-chasing: many short stints (< 18 months) among multiple roles.
    short = sum(1 for j in history if 0 < (j.get("duration_months", 0) or 0) < 18)
    if len(history) >= 3 and short >= 3:
        mult *= 0.7
        flags.append(f"{short} job changes under 18 months (tenure concern)")

    # Academia-only (professor/researcher current title, no production prose).
    cur = (profile.get("current_title", "") or "").lower()
    if ("professor" in cur or "researcher" in cur) and domain_detail.get("prose_hits", 0) < 1.0:
        mult *= 0.55
        flags.append("research/academic profile with little production evidence")

    return mult, flags


# --------------------------------------------------------------------------- #
# Top-level scoring
# --------------------------------------------------------------------------- #
def score_candidate(candidate: dict, semantic_sim: float) -> dict:
    """Return a dict with the final score and all component detail."""
    hp_reasons = honeypot_reasons(candidate)

    title_s, title_d = score_title(candidate)
    domain_s, domain_d = score_domain(candidate, semantic_sim)
    exp_s, exp_d = score_experience(candidate)
    skills_s, skills_d = score_skills(candidate)
    product_s, product_d = score_product(candidate)
    loc_s, loc_d = score_location(candidate)
    edu_s, edu_d = score_education(candidate)

    base = (
        jd.WEIGHTS["domain"] * domain_s
        + jd.WEIGHTS["title"] * title_s
        + jd.WEIGHTS["skills"] * skills_s
        + jd.WEIGHTS["experience"] * exp_s
        + jd.WEIGHTS["product"] * product_s
        + jd.WEIGHTS["location"] * loc_s
        + jd.WEIGHTS["education"] * edu_s
    )

    beh_mult, beh_d = behavioral_multiplier(candidate)
    dq_mult, dq_flags = disqualifier_multiplier(candidate, domain_d)

    final = base * beh_mult * dq_mult
    if hp_reasons:
        final = 0.0  # honeypots forced out of contention

    return {
        "candidate_id": candidate["candidate_id"],
        "score": final,
        "base": base,
        "components": {
            "title": title_s, "domain": domain_s, "experience": exp_s,
            "skills": skills_s, "product": product_s, "location": loc_s,
            "education": edu_s,
        },
        "behavioral_multiplier": beh_mult,
        "disqualifier_multiplier": dq_mult,
        "disqualifier_flags": dq_flags,
        "honeypot_reasons": hp_reasons,
        "detail": {**title_d, **domain_d, **exp_d, **skills_d, **product_d,
                   **loc_d, **edu_d, **beh_d},
    }
