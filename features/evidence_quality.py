# ==========================================================
# features/evidence_quality.py
#
# Responsibility: Credibility assessment only.
#
# This module answers ONE question:
#   "Can we trust the evidence that exists in this profile?"
#
# It does NOT measure how much evidence exists (that is
# evidence_engine.py's job) and does NOT measure whether
# the profile is internally consistent (that is
# consistency_engine.py's job).
#
# It accepts the already-computed evidence dict so that
# locate_evidence() is never called more than once per
# candidate across the entire pipeline.
#
# Output: a single credibility_multiplier in [0.5, 1.0].
#   1.0 = evidence is fully credible
#   0.5 = strong signals of stuffing or fabrication
#
# The multiplier is applied additively (scaled to points)
# inside signal_fusion.py — it never collapses other scores
# by multiplication.
# ==========================================================


# Titles that are structurally non-technical.
# A candidate with one of these titles who also claims
# dense AI/retrieval skills but has zero career backing
# is a credibility risk.

NON_TECH_TITLES = {
    "hr manager",
    "human resources manager",
    "accountant",
    "content writer",
    "content creator",
    "operations manager",
    "customer support",
    "customer success manager",
    "sales manager",
    "sales executive",
    "marketing manager",
    "digital marketing manager",
    "recruiter",
    "talent acquisition",
    "business development manager",
    "business analyst",
    "project manager",
    "product marketing manager",
}

# Categories that constitute meaningful technical depth
# for the target JD (retrieval / ranking systems).
# "python" is excluded here because it is too common
# to be a credibility signal by itself.

TECHNICAL_CATEGORIES = (
    "retrieval",
    "ranking",
    "vector_db",
    "embeddings",
    "evaluation",
    "llm",
)


def calculate_credibility(candidate, evidence):
    """
    Assess the credibility of a candidate's claimed expertise.

    Parameters
    ----------
    candidate : Candidate
        The candidate dataclass (needs current_title).
    evidence : dict
        The evidence dict already produced by locate_evidence().
        Shape: { category: { "headline": [...], "summary": [...],
                              "skills": [...], "career": [...] } }

    Returns
    -------
    dict with keys:
        credibility_multiplier  float   [0.5, 1.0]
        career_evidence_count   int     # of technical categories
                                        backed by career descriptions
        skills_evidence_count   int     # of technical keyword matches
                                        in the skills section only
        credibility_flags       list    human-readable penalty reasons
    """

    title = candidate.current_title.lower().strip()

    # ----------------------------------------------------------
    # Signal 1: Career-backed technical evidence
    # How many of the target technical categories appear in
    # actual job descriptions (not just in skills)?
    # This is the strongest credibility indicator.
    # ----------------------------------------------------------

    career_evidence_count = sum(
        1
        for cat in TECHNICAL_CATEGORIES
        if evidence[cat]["career"]
    )

    # ----------------------------------------------------------
    # Signal 2: Skills-section keyword density
    # High count here with zero career backing is the
    # classic "course collector" or "keyword stuffer" pattern.
    # ----------------------------------------------------------

    skills_evidence_count = sum(
        len(evidence[cat]["skills"])
        for cat in TECHNICAL_CATEGORIES
    )

    # ----------------------------------------------------------
    # Credibility scoring
    # Start at 1.0 and apply penalties only.
    # Penalties are independent and do not stack into collapse.
    # ----------------------------------------------------------

    multiplier = 1.0
    flags = []

    # Penalty A — Non-technical title with rich AI skill claims
    # but zero career backing.
    # Pattern: recruiter who listed "FAISS, BM25, LLM" to game ATS.

    if (
        title in NON_TECH_TITLES
        and skills_evidence_count >= 5
        and career_evidence_count == 0
    ):
        multiplier -= 0.30
        flags.append("non_tech_title_with_unsupported_ai_skills")

    # Penalty B — AI course collector.
    # Pattern: candidate has listed 8+ AI/ML skill keywords
    # but none appear in any career description.
    # Title is irrelevant here — even a "software engineer"
    # can stuff skills without backing them in experience.

    if (
        skills_evidence_count >= 8
        and career_evidence_count == 0
    ):
        multiplier -= 0.20
        flags.append("skills_without_career_evidence")

    # Penalty C — Moderate skill claim with no career evidence.
    # Softer version: 4–7 AI skills listed, still zero career
    # backing. Less extreme than the collector pattern but
    # still a credibility concern.

    if (
        skills_evidence_count >= 4
        and career_evidence_count == 0
        and "skills_without_career_evidence" not in flags
    ):
        multiplier -= 0.10
        flags.append("limited_career_evidence_for_claimed_skills")

    # Floor: never go below 0.5.
    # We reduce trust but never zero out a candidate entirely
    # on credibility alone — the evidence signals still matter.

    multiplier = max(0.5, round(multiplier, 4))

    return {
        "credibility_multiplier":   multiplier,
        "career_evidence_count":    career_evidence_count,
        "skills_evidence_count":    skills_evidence_count,
        "credibility_flags":        flags,
    }
