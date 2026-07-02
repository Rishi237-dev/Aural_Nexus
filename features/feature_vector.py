# ==========================================================
# features/feature_vector.py
#
# Responsibility: Assemble all upstream module outputs into
# a single flat dict that downstream modules (signal_fusion,
# reasoning_generator) can read without re-computing anything.
#
# Design rules:
#   1. locate_evidence() is called ONCE here and passed to
#      every module that needs it. No module below this one
#      is allowed to call locate_evidence() independently.
#   2. All keys in the returned vector must match exactly
#      what the caller modules expect. No silent mismatches.
#   3. This module performs NO scoring logic of its own.
#      It is a pure assembler.
# ==========================================================

from features.evidence_locator import locate_evidence

from features.jd_feature_extractor import extract_jd_features

from features.evidence_engine import calculate_evidence_score

from features.evidence_quality import calculate_credibility

from risk.consistency_engine import calculate_consistency


def build_feature_vector(candidate):
    """
    Build a complete feature vector for one candidate.

    locate_evidence() is called once and the result is passed
    to every downstream module that requires it, so no module
    repeats the text scan.

    Returns
    -------
    dict — flat key/value map of all signals needed by
           signal_fusion and reasoning_generator.
    """

    # ----------------------------------------------------------
    # Step 1: Locate evidence once.
    # Every module that needs the evidence dict receives this
    # exact object. No re-scanning the candidate text.
    # ----------------------------------------------------------

    evidence = locate_evidence(candidate)

    # ----------------------------------------------------------
    # Step 2: Run all upstream modules.
    # Each module receives only what it needs.
    # ----------------------------------------------------------

    jd          = extract_jd_features(candidate)
    ev_result   = calculate_evidence_score(candidate, evidence)
    credibility = calculate_credibility(candidate, evidence)
    consistency = calculate_consistency(candidate, evidence)

    # ----------------------------------------------------------
    # Step 3: Assemble the flat vector.
    # Keys are grouped by source module with clear labels.
    # Every key used by signal_fusion.py must be present here.
    # ----------------------------------------------------------

    vector = {

        # ------------------------------------------------------
        # Candidate metadata
        # ------------------------------------------------------

        "candidate_id":
            candidate.candidate_id,

        "current_title":
            candidate.current_title,

        "years_experience":
            candidate.years_experience,

        # ------------------------------------------------------
        # JD feature scores
        # Raw keyword match counts from jd_feature_extractor.
        # These are COUNTS (0, 1, 2, …), not 0-100 scores.
        # signal_fusion.py normalises them to 0-100.
        # ------------------------------------------------------

        "retrieval_score":
            jd["retrieval_score"],

        "vector_db_score":
            jd["vector_db_score"],

        "embeddings_score":
            jd["embeddings_score"],

        "ranking_score":
            jd["ranking_score"],

        "evaluation_score":
            jd["evaluation_score"],

        "llm_score":
            jd["llm_score"],

        "python_score":
            jd["python_score"],

        # ------------------------------------------------------
        # Evidence strength  (evidence_engine)
        # evidence_score: 0-100 aggregate strength.
        # category_scores: per-category weighted strengths.
        # raw_evidence: the source dict for reasoning_generator.
        # ------------------------------------------------------

        "evidence_score":
            ev_result["evidence_score"],

        "category_scores":
            ev_result["category_scores"],

        "raw_evidence":
            evidence,           # already computed above

        # ------------------------------------------------------
        # Credibility  (evidence_quality)
        # credibility_multiplier: float in [0.5, 1.0]
        # Used additively in signal_fusion — NOT as a score
        # compressor.
        # ------------------------------------------------------

        "credibility_multiplier":
            credibility["credibility_multiplier"],

        "career_evidence_count":
            credibility["career_evidence_count"],

        "skills_evidence_count":
            credibility["skills_evidence_count"],

        "credibility_flags":
            credibility["credibility_flags"],

        # ------------------------------------------------------
        # Consistency  (consistency_engine)
        #
        # Key mapping from consistency_engine return dict:
        #   consistency_score       → overall 0-100
        #   risk_score              → 100 - consistency_score
        #   title_alignment_score   → title ↔ career backing
        #   career_alignment_score  → summary ↔ career backing
        #                             (named "summary_score" in
        #                              engine but aliased here for
        #                              downstream clarity)
        #   skill_alignment_score   → skills ↔ multi-source
        #   career_progression_score→ seniority vs. years
        #   timeline_score          → plausibility of claimed level
        #   risk_flags              → list of flag strings
        # ------------------------------------------------------

        "consistency_score":
            consistency["consistency_score"],

        "risk_score":
            consistency["risk_score"],

        "title_alignment_score":
            consistency["title_alignment_score"],

        "career_alignment_score":
            consistency["career_alignment_score"],

        "skill_alignment_score":
            consistency["skill_alignment_score"],

        "career_progression_score":
            consistency["career_progression_score"],

        "timeline_score":
            consistency["timeline_score"],

        "risk_flags":
            consistency["risk_flags"],

    }

    return vector
