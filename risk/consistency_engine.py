# ==========================================================
# risk/consistency_engine.py
#
# Responsibility: Internal profile consistency only.
#
# This module answers: "Does the evidence in this profile
# tell a coherent story about the candidate's background?"
#
# It does NOT measure how much evidence exists
# (evidence_engine.py) and does NOT detect credibility risks
# (evidence_quality.py).
#
# The evidence dict is passed in from feature_vector.py so
# that locate_evidence() is called only once per candidate.
#
# STATUS: Frozen. Do not change scoring logic unless a
# verified defect is found. Only the function signature
# has been updated to accept the pre-computed evidence dict.
# ==========================================================

from features.evidence_locator import (
    locate_evidence,
    count_evidence_sources,
)


# ==========================================================
# Technical Role Keywords
# ==========================================================

TECH_TITLE_KEYWORDS = {
    "engineer",
    "developer",
    "scientist",
    "architect",
    "platform",
    "backend",
    "frontend",
    "full stack",
    "software",
    "cloud",
    "devops",
    "ml",
    "machine learning",
    "ai",
    "search",
    "recommendation",
    "nlp",
    "data",
}

NON_TECH_TITLE_KEYWORDS = {
    "marketing",
    "sales",
    "accountant",
    "hr",
    "customer support",
    "operations",
    "recruiter",
    "business development",
    "content writer",
}


# ==========================================================
# Helper Functions
# ==========================================================

def _normalize(text):
    if not text:
        return ""
    return text.lower().strip()


def _career_text(candidate):
    return " ".join(
        job.get("description", "")
        for job in candidate.career_history
    ).lower()


def _summary_text(candidate):
    return _normalize(candidate.summary)


def _title_text(candidate):
    return _normalize(candidate.current_title)


def _contains_any(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return True
    return False


# ==========================================================
# Title <-> Career Alignment
# ==========================================================

def _title_alignment_score(candidate, evidence):

    title = _title_text(candidate)

    tech_title = _contains_any(title, TECH_TITLE_KEYWORDS)
    non_tech_title = _contains_any(title, NON_TECH_TITLE_KEYWORDS)

    technical_categories = (
        "retrieval",
        "ranking",
        "vector_db",
        "embeddings",
        "evaluation",
        "llm",
        "python",
    )

    career_support = sum(
        1
        for category in technical_categories
        if evidence[category]["career"]
    )

    if tech_title:
        if career_support >= 4:
            return 25
        if career_support >= 2:
            return 20
        if career_support >= 1:
            return 15
        return 8

    if non_tech_title:
        if career_support >= 3:
            return 8
        return 2

    if career_support >= 3:
        return 15
    if career_support >= 1:
        return 10
    return 5


# ==========================================================
# Summary <-> Career Alignment
# ==========================================================

def _summary_alignment_score(candidate, evidence):

    score = 0

    important_categories = (
        "retrieval",
        "ranking",
        "vector_db",
        "embeddings",
        "evaluation",
        "python",
    )

    for category in important_categories:

        summary_has = bool(evidence[category]["summary"])
        career_has  = bool(evidence[category]["career"])

        if summary_has and career_has:
            score += 3
        elif summary_has and not career_has:
            score -= 1

    return max(0, min(score, 20))


# ==========================================================
# Skills <-> Career Alignment
# ==========================================================

def _skill_alignment_score(candidate, evidence):

    important_categories = (
        "retrieval",
        "ranking",
        "vector_db",
        "embeddings",
        "evaluation",
        "python",
    )

    score = 0

    for category in important_categories:

        source_count = count_evidence_sources(
            evidence[category]
        )

        if source_count >= 3:
            score += 4
        elif source_count == 2:
            score += 3
        elif source_count == 1:
            score += 2

    return min(score, 20)


# ==========================================================
# Career Progression Plausibility
# ==========================================================

def _career_progression_score(candidate):

    title = _title_text(candidate)
    years = candidate.years_experience

    score = 15

    if   "principal" in title and years < 8:
        score -= 10
    elif "staff"     in title and years < 6:
        score -= 8
    elif "lead"      in title and years < 5:
        score -= 5
    elif "senior"    in title and years < 3:
        score -= 4
    elif "manager"   in title and years < 2:
        score -= 4

    return max(score, 0)


# ==========================================================
# Timeline Plausibility
# ==========================================================

def _timeline_score(candidate):

    title = _title_text(candidate)
    years = candidate.years_experience

    score = 20

    if   "principal" in title and years < 8:
        score -= 20
    elif "staff"     in title and years < 6:
        score -= 15
    elif "lead"      in title and years < 5:
        score -= 10
    elif "senior"    in title and years < 3:
        score -= 8
    elif "manager"   in title and years < 2:
        score -= 6

    return max(score, 0)


# ==========================================================
# Risk Flags
# ==========================================================

def _risk_flags(
    title_score,
    summary_score,
    skill_score,
    progression_score,
    timeline_score,
):
    flags = []

    if title_score     <= 8:
        flags.append("title_career_mismatch")
    if summary_score   <= 6:
        flags.append("summary_career_mismatch")
    if skill_score     <= 6:
        flags.append("skills_without_support")
    if progression_score <= 6:
        flags.append("inconsistent_career_progression")
    if timeline_score  <= 8:
        flags.append("implausible_seniority")

    return flags


# ==========================================================
# Main API
# ==========================================================

def calculate_consistency(candidate, evidence=None):
    """
    Assess the internal consistency of a candidate's profile.

    Parameters
    ----------
    candidate : Candidate
        The candidate dataclass.
    evidence : dict, optional
        Pre-computed output of locate_evidence(). If provided,
        locate_evidence() is not called again.
        Pass this from feature_vector.py to avoid re-scanning.

    Returns
    -------
    dict with keys:
        consistency_score           int     0-100
        risk_score                  int     100 - consistency_score
        title_alignment_score       int     title ↔ career backing
        career_alignment_score      int     summary ↔ career backing
        skill_alignment_score       int     skills multi-source spread
        career_progression_score    int     seniority vs years
        timeline_score              int     timeline plausibility
        risk_flags                  list    flag strings
    """

    if evidence is None:
        evidence = locate_evidence(candidate)

    title_score       = _title_alignment_score(candidate, evidence)
    summary_score     = _summary_alignment_score(candidate, evidence)
    skill_score       = _skill_alignment_score(candidate, evidence)
    progression_score = _career_progression_score(candidate)
    timeline_score    = _timeline_score(candidate)

    consistency_score = max(0, min(
        title_score +
        summary_score +
        skill_score +
        progression_score +
        timeline_score,
        100,
    ))

    risk_score = 100 - consistency_score

    risk_flags = _risk_flags(
        title_score,
        summary_score,
        skill_score,
        progression_score,
        timeline_score,
    )

    return {
        "consistency_score":            consistency_score,
        "risk_score":                   risk_score,
        "title_alignment_score":        title_score,
        "career_alignment_score":       summary_score,
        "skill_alignment_score":        skill_score,
        "career_progression_score":     progression_score,
        "timeline_score":               timeline_score,
        "risk_flags":                   risk_flags,
    }
