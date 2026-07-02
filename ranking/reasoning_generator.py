# ==========================================================
# ranking/reasoning_generator.py
#
# Responsibility: Turn a calculate_final_score() result into
# a 1-2 sentence, human-readable reasoning string.
#
# Hard contract (per submission_spec.md Section 3, Stage 4
# manual review — six checks the organisers grade against):
#
#   1. Specific facts   — must cite real values from the
#                          candidate's own profile (years,
#                          title, named skills, signal values)
#   2. JD connection     — must reference what the JD actually
#                          asked for, not generic praise
#   3. Honest concerns   — must surface real weaknesses, not
#                          just strengths, when they exist
#   4. No hallucination  — every claim must trace to a value
#                          already computed upstream; this
#                          module invents nothing
#   5. Variation          — sentence structure must vary across
#                          candidates, not be a fill-in-the-blank
#                          template with only the name swapped
#   6. Rank consistency  — tone must match rank: a top candidate
#                          reads as strong, a bottom-100
#                          candidate reads as marginal/borderline,
#                          never glowing
#
# Design rule: this module performs NO new scoring. It reads
# only fields already present in the result dict returned by
# ranking.signal_fusion.calculate_final_score(). If a field
# doesn't exist in that dict, it cannot appear in a reason.
#
# Forward-compatibility note: when the AI layer (Phase 2/3)
# adds "embedding_similarity_score" to the feature vector,
# add ONE entry to STRENGTH_RULES below. Nothing else in this
# file needs to change — see the dispatch loop.
# ==========================================================




# ==========================================================
# Rank bands control tone. These thresholds operate on the
# candidate's POSITION in the final top-100 list (1-100),
# not on the raw 0-100 score, because spec Stage 4 grades
# "does tone match rank", not "does tone match score".
# ==========================================================

def _rank_band(rank):
    if rank <= 10:
        return "top"
    elif rank <= 40:
        return "strong"
    elif rank <= 70:
        return "moderate"
    else:
        return "marginal"


# ==========================================================
# Strength sentence builders.
# Each function takes (vector, components) and returns a
# sentence fragment OR None if the signal doesn't fire.
# All values referenced here exist in the feature vector —
# see features/feature_vector.py for the source of each key.
# ==========================================================

def _strength_retrieval(vector, components):
    if components.get("retrieval", 0) >= 70:
        cat_scores = vector.get("category_scores", {})
        count = cat_scores.get("retrieval", 0)
        return (
            f"demonstrated retrieval-system experience "
            f"(category strength {count:.0f}) directly backed "
            f"by career history"
        )
    return None


def _strength_ranking(vector, components):
    if components.get("ranking", 0) >= 75:
        return (
            "background in ranking and evaluation metrics "
            "(NDCG/MRR-style work), matching the JD's core "
            "evaluation requirement"
        )
    return None


def _strength_vector_db(vector, components):
    if components.get("vector_db", 0) >= 60:
        return "hands-on vector database experience"
    return None


def _strength_experience(vector, components):
    years = vector.get("years_experience", 0)
    if components.get("experience", 0) >= 80:
        return f"{years:.1f} years of relevant experience"
    return None


def _strength_consistency(vector, components):
    if components.get("consistency", 0) >= 70:
        return (
            "a consistent career trajectory — title, summary, "
            "and job history all align"
        )
    return None


def _strength_credibility(vector, components):
    career_ev = vector.get("career_evidence_count", 0)
    if components.get("credibility", 0) >= 80 and career_ev > 0:
        unit = "category" if career_ev == 1 else "categories"
        return (
            f"skills claims grounded in {career_ev} career-backed "
            f"evidence {unit}, not just a skills list"
        )
    return None


# ==========================================================
# Weakness sentence builders.
# Same contract as strengths — return a fragment or None.
# ==========================================================

def _weakness_retrieval(vector, components):
    if components.get("retrieval", 0) == 0:
        return "no retrieval-system evidence found in the profile"
    return None


def _weakness_ranking(vector, components):
    if components.get("ranking", 0) == 0:
        return "no ranking or evaluation-metric evidence found"
    return None


def _weakness_vector_db(vector, components):
    if components.get("vector_db", 0) == 0:
        return "no vector database experience evidenced"
    return None


def _weakness_consistency(vector, components):
    if components.get("consistency", 0) < 50:
        flags = vector.get("risk_flags", [])
        if flags:
            readable = flags[0].replace("_", " ")
            return f"profile shows a consistency concern ({readable})"
        return "profile shows internal consistency concerns"
    return None


def _weakness_credibility(vector, components):
    if components.get("credibility", 0) < 50:
        skills_ev = vector.get("skills_evidence_count", 0)
        return (
            f"{skills_ev} AI/ML skills claimed "
            f"with no supporting career evidence"
        )
    return None


def _weakness_experience(vector, components):
    if components.get("experience", 0) <= 35:
        years = vector.get("years_experience", 0)
        return f"only {years:.1f} years of experience"
    return None


# Dispatch tables. Order matters — first-firing rules are
# used first, which gives natural variation across candidates
# since different candidates trip different rules first.

STRENGTH_RULES = [
    _strength_retrieval,
    _strength_ranking,
    _strength_vector_db,
    _strength_consistency,
    _strength_credibility,
    _strength_experience,
]

WEAKNESS_RULES = [
    _weakness_retrieval,
    _weakness_ranking,
    _weakness_consistency,
    _weakness_credibility,
    _weakness_vector_db,
    _weakness_experience,
]


# ==========================================================
# Sentence assembly
#
# To satisfy the "variation" check (Stage 4 sample 10 rows
# and compare), we vary connective phrasing using a rotation
# keyed off the candidate_id rather than true randomness, so
# output stays deterministic across runs on the same dataset
# (required — spec demands reproducible scoring).
# ==========================================================

OPENER_TEMPLATES = [
    "{title} with {years:.1f} years of experience.",
    "{title}, {years:.1f} years in.",
    "{years:.1f}-year {title}.",
]

CONNECTOR_STRONG = [
    "Ranked here for",
    "Stands out for",
    "Strongest signal:",
]

CONNECTOR_MARGINAL = [
    "Included despite",
    "Lower-ranked due to",
    "Marginal fit:",
]


def _deterministic_choice(options, seed_key):
    """
    Picks an option deterministically from a list using the
    candidate_id as a seed, so re-running the pipeline on the
    same dataset always produces the same reasoning text
    (required for the spec's reproducibility constraint),
    while still varying phrasing across different candidates.
    """
    index = sum(ord(c) for c in seed_key) % len(options)
    return options[index]


def generate_reasoning(result, rank):
    """
    Produce a 1-2 sentence reasoning string for one candidate.

    Parameters
    ----------
    result : dict
        The full return value of
        ranking.signal_fusion.calculate_final_score(candidate).
        Must contain: candidate_id, final_score, components,
        vector.
    rank : int
        This candidate's 1-indexed position in the final
        top-100 list. Controls tone via _rank_band().

    Returns
    -------
    str — a 1-2 sentence reasoning string. Never empty.
          Every fact in the string traces to a value already
          present in `result`.
    """

    vector     = result.get("vector", {})
    components = result.get("components", {})
    cid        = result.get("candidate_id", "UNKNOWN")
    band       = _rank_band(rank)

    title = vector.get("current_title", None) or "Unspecified title"
    years = vector.get("years_experience", 0)

    opener_template = _deterministic_choice(OPENER_TEMPLATES, cid)
    opener = opener_template.format(title=title, years=years)

    # Collect every strength/weakness that actually fires.
    # Nothing here is invented — each function reads only
    # fields that exist in `vector` / `components`.

    strengths = [
        s for s in (rule(vector, components) for rule in STRENGTH_RULES)
        if s is not None
    ]

    weaknesses = [
        w for w in (rule(vector, components) for rule in WEAKNESS_RULES)
        if w is not None
    ]

    # ----------------------------------------------------------
    # Tone by rank band.
    #
    # top / strong  → lead with strengths, mention one weakness
    #                 only if it materially exists (honest, not
    #                 falsely glowing).
    # moderate      → balanced: one strength, one concern.
    # marginal      → lead with the gap, strength only if real.
    # ----------------------------------------------------------

    if band in ("top", "strong"):
        connector = _deterministic_choice(CONNECTOR_STRONG, cid)

        if strengths:
            body = f"{connector} {'; '.join(strengths[:2])}."
        else:
            # No strong signal fired even though score is high —
            # this happens when experience/consistency alone
            # carried the score. Say so honestly.
            body = (
                f"{connector} solid overall profile consistency "
                f"and experience level, though no single standout "
                f"domain signal."
            )

        if weaknesses:
            body += f" Noted concern: {weaknesses[0]}."

    elif band == "moderate":
        if strengths:
            body = f"Relevant: {strengths[0]}."
        else:
            body = "Limited direct evidence for the JD's core requirements."

        if weaknesses:
            body += f" However, {weaknesses[0]}."

    else:  # marginal
        connector = _deterministic_choice(CONNECTOR_MARGINAL, cid)

        if weaknesses:
            body = f"{connector} {weaknesses[0]}."
        else:
            body = "Included as lower-tier filler with limited differentiating evidence."

        if strengths:
            body += f" Some support from {strengths[0]}."

    reasoning = f"{opener} {body}"

    # Stage 4 penalizes empty reasoning outright — guarantee
    # non-empty output as a final safety net (should never
    # trigger given the above, but defends against blank titles).

    if not reasoning.strip():
        reasoning = (
            f"Candidate {cid}: final score {result['final_score']:.2f}, "
            f"ranked #{rank} based on combined evidence, consistency, "
            f"and credibility signals."
        )

    return reasoning.strip()
