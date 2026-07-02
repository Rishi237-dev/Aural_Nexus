# ==========================================================
# ranking/signal_fusion.py
#
# Responsibility: Compute the final ranking score.
#
# Design contract:
#   - The final score is a PURE ADDITIVE WEIGHTED SUM.
#   - Consistency and credibility are additive bonus
#     components, NOT multipliers.
#   - No score compression. A candidate with strong evidence
#     gets a high score even if they are a junior.
#   - Every component is normalised to 0-100 before weighting
#     so that weights are directly interpretable as "how many
#     points can this component contribute at maximum?"
#   - Component weights sum to 100. Final score is in [0, 100].
#
# Why no multipliers?
#   Multiplying by fractional modifiers (e.g. 0.6 × 0.7 = 0.42)
#   collapses all scores toward zero and destroys the rank
#   separation needed to produce a meaningful Top 100.
#   Additive components preserve that separation.
#
# Score anatomy (max points per component):
#   Retrieval experience       25 pts  ← core JD requirement
#   Ranking/evaluation         18 pts  ← core JD requirement
#   Vector DB experience       10 pts  ← supporting JD skill
#   Years of experience        13 pts  ← seniority signal
#   Profile consistency        14 pts  ← credibility of history
#   Evidence credibility       10 pts  ← trust in claimed skills
#   Retrieval ext. score       10 pts  ← pre-generated dense+BM25+RRF
#                             -------
#   Total                     100 pts
# ==========================================================

from features.feature_vector import build_feature_vector


# ==========================================================
# Component weight table.
# Weights represent maximum points that component can add.
# They must sum to 100.
# ==========================================================

COMPONENT_WEIGHTS = {
    "retrieval":        25,   # core JD requirement (text-based)
    "ranking":          18,   # ranking + evaluation metrics
    "vector_db":        10,   # supporting JD skill
    "experience":       13,   # seniority signal
    "consistency":      14,   # profile credibility
    "credibility":      10,   # evidence trust
    "retrieval_ext":    10,   # pre-generated dense+BM25+RRF score
}

assert sum(COMPONENT_WEIGHTS.values()) == 100, (
    "COMPONENT_WEIGHTS must sum to 100"
)


# ==========================================================
# Component normalisers
# Each function returns a value in [0, 100] representing
# how well the candidate performs on that dimension.
# ==========================================================

def _retrieval_component(vector):
    """
    Measures retrieval-specific evidence depth.

    Uses the raw JD keyword count (retrieval_score) from
    jd_feature_extractor as a proxy for domain coverage,
    then scales it using the career-backed evidence count
    from credibility assessment to reward candidates where
    the experience is grounded in actual job history.

    Levels are intentionally non-linear: a single retrieval
    mention is meaningfully different from three, but the
    gap from 3 to 5 is not proportional.
    """

    keyword_count       = vector["retrieval_score"]
    career_ev_count     = vector["career_evidence_count"]

    # Base score from keyword presence
    if keyword_count >= 3:
        base = 100
    elif keyword_count == 2:
        base = 70
    elif keyword_count == 1:
        base = 40
    else:
        base = 0

    # Bonus for career-backed depth.
    # A candidate with retrieval keywords AND career evidence
    # across multiple categories is more valuable.
    # Cap total at 100.
    if base > 0 and career_ev_count >= 3:
        base = min(base + 15, 100)
    elif base > 0 and career_ev_count >= 1:
        base = min(base + 5, 100)

    return base


def _ranking_component(vector):
    """
    Measures ranking and evaluation domain coverage.

    Combines ranking_score and evaluation_score because
    a candidate who understands both ranking systems and
    evaluation metrics (NDCG, MRR) is more complete than
    one who only mentions ranking keywords in passing.
    """

    combined = vector["ranking_score"] + vector["evaluation_score"]

    if combined >= 3:
        return 100
    elif combined == 2:
        return 75
    elif combined == 1:
        return 50
    else:
        return 0


def _vector_db_component(vector):
    """
    Measures vector database experience.

    A secondary JD requirement. Single mention is already
    notable given how specialised these tools are.
    """

    score = vector["vector_db_score"]

    if score >= 2:
        return 100
    elif score == 1:
        return 60
    else:
        return 0


def _experience_component(years):
    """
    Converts years of experience to a 0-100 score.

    The curve is intentionally non-linear at the low end —
    the difference between 0 and 3 years matters more than
    the difference between 8 and 12 years for this role.
    """

    if years >= 8:
        return 100
    elif years >= 5:
        return 80
    elif years >= 3:
        return 60
    elif years >= 1:
        return 35
    else:
        return 15


def _consistency_component(vector):
    """
    Converts the consistency_score (0-100) to 0-100.

    consistency_score is already 0-100 from the engine,
    so this is a pass-through. Kept as a named function
    for explainability clarity.
    """

    return vector["consistency_score"]


def _credibility_component(vector):
    """
    Converts the credibility_multiplier [0.5, 1.0] to 0-100.

    Maps the [0.5, 1.0] multiplier range linearly onto
    [0, 100] so it contributes as an additive component
    rather than collapsing all other scores.

    0.5 → 0 pts (minimum credibility)
    1.0 → 100 pts (full credibility)
    """

    multiplier = vector["credibility_multiplier"]
    return round((multiplier - 0.5) / 0.5 * 100, 2)


def _retrieval_ext_component(
    raw_score: float,
    score_min: float,
    score_max: float,
) -> float:
    """
    Normalise the pre-generated final_retrieval_score to [0, 100].

    The raw score is a Reciprocal-Rank Fusion (RRF) value weighted by
    a redrob multiplier. The normalisation bounds (score_min, score_max)
    are derived dynamically from retrieval_results.json at load time
    in main.py and passed in here — no hardcoded constants.

    Normalisation formula:
        normalised = (raw - score_min) / (score_max - score_min) × 100

    Values below score_min clamp to 0; values above score_max clamp to 100.
    A candidate absent from retrieval_results.json receives 0.0 (the
    default passed in from main.py), which clamps to 0.
    """
    if raw_score <= score_min:
        return 0.0
    if raw_score >= score_max:
        return 100.0
    return round(
        (raw_score - score_min)
        / (score_max - score_min)
        * 100,
        2,
    )


# ==========================================================
# Explanation helpers
# ==========================================================

def _build_strengths(vector, components):
    """
    Produce a list of human-readable strength statements
    derived purely from computed component values.
    No hallucination — every statement maps to a number.
    """

    strengths = []

    if components["retrieval"] >= 70:
        strengths.append(
            "Strong retrieval system experience in career history"
        )
    elif components["retrieval"] >= 40:
        strengths.append(
            "Some retrieval exposure mentioned in profile"
        )

    if components["ranking"] >= 75:
        strengths.append(
            "Solid ranking and evaluation metrics background (NDCG/MRR)"
        )
    elif components["ranking"] >= 50:
        strengths.append(
            "Ranking or evaluation experience present"
        )

    if components["vector_db"] >= 60:
        strengths.append(
            "Hands-on vector database experience (FAISS/Pinecone/Milvus etc.)"
        )

    if components["experience"] >= 80:
        strengths.append(
            f"Experienced professional ({vector['years_experience']:.0f}+ years)"
        )

    if components["consistency"] >= 70:
        strengths.append(
            "Profile is internally consistent — title, summary, and career align"
        )

    if components["credibility"] >= 80:
        strengths.append(
            "Evidence is well-grounded in career descriptions, not just skill tags"
        )

    if components["retrieval_ext"] >= 60:
        strengths.append(
            "Ranked highly by the retrieval model (dense + BM25 + RRF signal)"
        )

    return strengths


def _build_weaknesses(vector, components):
    """
    Produce a list of human-readable weakness / risk statements.
    """

    weaknesses = []

    if components["retrieval"] == 0:
        weaknesses.append("No retrieval system evidence found")

    if components["ranking"] == 0:
        weaknesses.append("No ranking or evaluation evidence found")

    if components["vector_db"] == 0:
        weaknesses.append("No vector database experience found")

    if components["consistency"] < 50:
        weaknesses.append("Profile shows internal inconsistencies")

    if components["credibility"] < 50:
        weaknesses.append(
            "Claimed skills appear unsupported by career history"
        )

    if components["retrieval_ext"] < 20:
        weaknesses.append(
            "Low retrieval model score — may not surface well in semantic search"
        )

    # Surface all flags for full transparency
    for flag in vector.get("risk_flags", []):
        if flag not in weaknesses:
            weaknesses.append(flag)

    for flag in vector.get("credibility_flags", []):
        if flag not in weaknesses:
            weaknesses.append(flag)

    return weaknesses


# ==========================================================
# Main API
# ==========================================================

def calculate_final_score(
    candidate,
    retrieval_score: float = 0.0,
    retrieval_ext_min: float = 0.0,
    retrieval_ext_max: float = 1.0,
):
    """
    Compute the final ranking score for one candidate.

    The score is a weighted additive sum of seven components,
    each normalised to 0-100. The final score is in [0, 100].

    Parameters
    ----------
    candidate : Candidate
        The candidate dataclass.
    retrieval_score : float
        The pre-generated final_retrieval_score from
        retrieval/artifacts/retrieval_results.json.
        Defaults to 0.0 if the candidate is absent from that file.
    retrieval_ext_min : float
        The minimum final_retrieval_score observed across the full
        candidate pool. Computed dynamically in main.py.
    retrieval_ext_max : float
        The maximum final_retrieval_score observed across the full
        candidate pool. Computed dynamically in main.py.

    Returns
    -------
    dict with keys:
        candidate_id    str
        final_score     float   0-100, rounded to 4 dp
        components      dict    per-component 0-100 values
        weights         dict    the weight applied to each
        strengths       list    human-readable positive signals
        weaknesses      list    human-readable risk signals
        vector          dict    the full feature vector
                                (for reasoning_generator)
    """

    vector = build_feature_vector(candidate)

    # ----------------------------------------------------------
    # Compute each component (all 0-100)
    # ----------------------------------------------------------

    components = {
        "retrieval":      _retrieval_component(vector),
        "ranking":        _ranking_component(vector),
        "vector_db":      _vector_db_component(vector),
        "experience":     _experience_component(
                              vector["years_experience"]
                          ),
        "consistency":    _consistency_component(vector),
        "credibility":    _credibility_component(vector),
        "retrieval_ext":  _retrieval_ext_component(
                              retrieval_score,
                              retrieval_ext_min,
                              retrieval_ext_max,
                          ),
    }

    # ----------------------------------------------------------
    # Weighted additive sum
    # final_score = Σ (component_i × weight_i / 100)
    # Dividing weight by 100 converts "max points" to a
    # fractional weight so the sum stays in [0, 100].
    # ----------------------------------------------------------

    final_score = sum(
        components[name] * COMPONENT_WEIGHTS[name] / 100
        for name in COMPONENT_WEIGHTS
    )

    final_score = round(final_score, 4)

    # ----------------------------------------------------------
    # Explanation
    # ----------------------------------------------------------

    strengths  = _build_strengths(vector, components)
    weaknesses = _build_weaknesses(vector, components)

    return {
        "candidate_id": vector["candidate_id"],
        "final_score":  final_score,
        "components":   components,
        "weights":      COMPONENT_WEIGHTS,
        "strengths":    strengths,
        "weaknesses":   weaknesses,
        "vector":       vector,       # passed to reasoning_generator
    }
