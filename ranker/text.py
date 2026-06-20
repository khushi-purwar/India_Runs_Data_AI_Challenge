"""Build text representations of a candidate.

We deliberately separate two kinds of text:

* ``prose`` - the summary plus every career-history description. This is the
  hard-to-fake evidence of what the person actually did.
* ``skills_text`` - the comma-separated skills array. Easy to stuff, so it is
  trusted far less.

The semantic layer embeds ``prose`` (weighted) plus a light touch of the rest.
"""
from __future__ import annotations


def career_prose(candidate: dict) -> str:
    """Concatenate the summary and all career-history descriptions/titles."""
    parts: list[str] = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("company", ""))
        parts.append(job.get("industry", ""))
        parts.append(job.get("description", ""))
    return "  ".join(p for p in parts if p)


def skills_text(candidate: dict) -> str:
    return ", ".join(s.get("name", "") for s in candidate.get("skills", []))


def full_text(candidate: dict) -> str:
    """Everything, used only for the semantic vectoriser."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        career_prose(candidate),
        skills_text(candidate),
    ]
    for edu in candidate.get("education", []):
        parts.append(edu.get("degree", ""))
        parts.append(edu.get("field_of_study", ""))
    return "  ".join(p for p in parts if p)
