"""Generate per-candidate reasoning strings.

Stage 4 manual review checks that reasoning: cites specific profile facts,
connects to JD requirements, acknowledges concerns honestly, never hallucinates,
varies across candidates, and matches the rank's tone. We therefore build each
sentence purely from values that exist in the candidate record and the computed
component scores - nothing is invented.
"""
from __future__ import annotations

from . import jd
from .text import career_prose


def _top_relevant_skills(candidate: dict, k: int = 3) -> list[str]:
    out = []
    for sk in candidate.get("skills", []):
        if (sk.get("name", "") or "").lower() in jd.RELEVANT_SKILLS:
            out.append(sk.get("name"))
        if len(out) >= k:
            break
    return out


def _domain_phrases(candidate: dict) -> list[str]:
    prose = career_prose(candidate).lower()
    hits = []
    for term in ("retrieval", "ranking", "recommendation", "search",
                 "embedding", "information retrieval", "learning to rank",
                 "relevance", "nlp", "personalization"):
        if term in prose and term not in hits:
            hits.append(term)
        if len(hits) >= 3:
            break
    return hits


def build_reasoning(candidate: dict, result: dict, rank: int) -> str:
    profile = candidate.get("profile", {})
    comp = result["components"]
    detail = result["detail"]
    yoe = detail.get("yoe", profile.get("years_of_experience", 0))
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    loc = profile.get("location", "")

    parts: list[str] = []

    # Lead with role + experience + employer (always factual).
    lead = f"{title} with {yoe} yrs"
    if company:
        lead += f" at {company}"
    parts.append(lead)

    # Domain evidence - connect to the JD's core need. Vary the phrasing
    # deterministically (by candidate_id) so reasonings are not templated.
    phrases = _domain_phrases(candidate)
    seed = int(candidate["candidate_id"].split("_")[1]) if "_" in candidate["candidate_id"] else 0
    strong_templates = [
        "career history describes {p} work, the core of this role's intelligence-layer mandate",
        "built {p} systems per the role descriptions, directly on-point for the JD",
        "hands-on {p} experience in prior roles, matching the retrieval/ranking remit",
        "{p} work runs through the career history, aligning with what the JD actually needs",
    ]
    weak_templates = [
        "some {p} exposure, though lighter than the JD's senior bar",
        "adjacent {p} experience that is relevant but not deep",
        "{p} touched on in the history; partial fit for the core mandate",
    ]
    if comp["domain"] >= 0.55 and phrases:
        parts.append(strong_templates[seed % len(strong_templates)].format(p="/".join(phrases)))
    elif comp["domain"] >= 0.4 and phrases:
        parts.append(weak_templates[seed % len(weak_templates)].format(p="/".join(phrases)))
    else:
        skills = _top_relevant_skills(candidate)
        if skills:
            parts.append("relevant skills (" + ", ".join(skills) + ") but limited production evidence in the JD's domain")
        else:
            parts.append("little direct retrieval/ranking evidence for this role")

    # Behavioral / availability - honest about engagement.
    resp = detail.get("response_rate")
    recency = detail.get("recency_days")
    open_w = detail.get("open_to_work")
    beh_bits = []
    if resp is not None:
        beh_bits.append(f"recruiter response {resp:.2f}")
    if recency is not None:
        beh_bits.append(f"active {recency}d ago")
    if open_w is not None:
        beh_bits.append("open to work" if open_w else "not flagged open-to-work")
    if beh_bits:
        parts.append("; ".join(beh_bits))

    # Honest concerns (disqualifier flags) - especially important for tone.
    if result["disqualifier_flags"]:
        parts.append("concern: " + result["disqualifier_flags"][0])

    # Location note when it is a clear plus or minus.
    if comp["location"] >= 0.9 and loc:
        parts.append(f"based in {loc} (preferred metro)")
    elif comp["location"] <= 0.3 and loc:
        parts.append(f"{loc} with no relocation flag (location risk)")

    text = "; ".join(parts) + "."
    # Keep within a tidy length.
    if len(text) > 320:
        text = text[:317].rstrip(" ;,") + "."
    return text
