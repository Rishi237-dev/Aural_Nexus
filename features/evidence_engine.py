# ==========================================================
# features/evidence_engine.py
#
# Responsibility: Evidence STRENGTH measurement only.
#
# This module answers: "How much evidence exists, and how
# strongly is it distributed across profile sections?"
#
# It does NOT assess credibility (evidence_quality.py) and
# does NOT check internal consistency (consistency_engine.py).
#
# The evidence dict is passed in from feature_vector.py so
# that locate_evidence() is only called once per candidate
# across the entire pipeline.
#
# Source weights reflect how credible each profile section
# is as evidence:
#   career      > summary > headline > skills
# because career descriptions are harder to fake than a
# skills list and more specific than a summary paragraph.
# ==========================================================

from features.evidence_locator import locate_evidence


# Weight per profile section.
# Career descriptions are the gold standard; skills lists
# are the weakest signal (easy to append without backing).

SOURCE_WEIGHTS = {
    "career":   4,
    "summary":  3,
    "headline": 2,
    "skills":   1,
}

# Weight per JD category.
# Reflects how central each category is to the target role.
# Retrieval and ranking are the core JD requirements;
# LLM fine-tuning is a nice-to-have, not the primary need.

CATEGORY_WEIGHTS = {
    "retrieval":    25,
    "ranking":      25,
    "evaluation":   20,
    "vector_db":    15,
    "embeddings":   10,
    "llm":           5,
}

# "python" is intentionally excluded from CATEGORY_WEIGHTS.
# It is nearly universal and contributes noise rather than
# signal when weighted the same as domain-specific skills.
# It is still captured in jd_feature_extractor for use as
# a hygiene check in signal_fusion.


def _category_evidence_score(category_evidence):
    """
    Compute a raw strength score for one evidence category.

    Each source section that has at least one match
    contributes: source_weight × number_of_matches.
    Multiple matches in the same source section add up,
    but are capped indirectly by the overall min(score, 100)
    in calculate_evidence_score().
    """
    score = 0

    for source, matches in category_evidence.items():
        if matches:
            score += SOURCE_WEIGHTS[source] * len(matches)

    return score


def calculate_evidence_score(candidate, evidence=None):
    """
    Compute the aggregate evidence strength score.

    Parameters
    ----------
    candidate : Candidate
        The candidate dataclass. Only used if evidence is None.
    evidence : dict, optional
        Pre-computed output of locate_evidence(). If provided,
        locate_evidence() is not called again.
        Pass this from feature_vector.py to avoid re-scanning.

    Returns
    -------
    dict with keys:
        evidence_score    int     0-100 aggregate strength
        category_scores   dict    per-category weighted scores
    """

    if evidence is None:
        evidence = locate_evidence(candidate)

    category_scores = {}
    total_score = 0

    for category, weight in CATEGORY_WEIGHTS.items():

        raw_score = _category_evidence_score(
            evidence[category]
        )

        weighted_score = raw_score * weight

        category_scores[category] = weighted_score
        total_score += weighted_score

    evidence_strength = min(total_score, 100)

    return {
        "evidence_score":   evidence_strength,
        "category_scores":  category_scores,
    }
