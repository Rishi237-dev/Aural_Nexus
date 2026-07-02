"""
candidate_document_builder.py
==============================
Builds the two retrieval documents required for the hybrid retrieval stage:

    1. Semantic Document  — natural-language text for E5 embedding.
    2. Lexical Document   — dense token list optimized for BM25 matching.

Both documents are derived from a single filtered-candidate JSON record
(the schema produced by the upstream Candidate Gate stage).

Excluded fields (must never leak into embeddings or BM25 index):
    - anonymized_name
    - location / country
    - current_company_size
    - salary (expected_salary_range_inr_lpa)
    - recruiter popularity (saved_by_recruiters_30d, search_appearance_30d,
      recruiter_response_rate, offer_acceptance_rate)
    - profile_views_received_30d
    - connection_count
    - endorsements_received / per-skill endorsements
    - linkedin_connected

These are excluded because they are either irrelevant to candidate-JD
relevance (logistics, popularity) or risk introducing bias into the
similarity computation.

This module performs NO embedding, NO indexing, NO scoring. It only
transforms one candidate record into two plain text strings plus a
small set of structured passthrough fields needed downstream
(candidate_id, redrob signals for the later multiplier stage).

CPU-only. No external APIs. Streaming-friendly: operates on one
candidate dict at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. OUTPUT DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class CandidateRetrievalDocument:
    """
    Container for the two retrieval documents plus the minimal structured
    metadata needed by later stages (embedding, BM25, redrob adjustment).

    Attributes
    ----------
    candidate_id : str
        Unique identifier, passed through unchanged.
    semantic_document : str
        Natural-language text for E5 embedding.
    lexical_document : str
        Dense space-separated token list for BM25.
    redrob_signals : dict[str, Any]
        Only the objective availability signals allowed by the architecture
        (profile_completeness_score, open_to_work_flag, last_active_date,
        interview_completion_rate, verified_email, verified_phone).
        Stored here so the redrob_adjustment module doesn't need to re-parse
        the raw candidate JSON later.
    """
    candidate_id: str
    semantic_document: str
    lexical_document: str
    redrob_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (used when persisting alongside embeddings)."""
        return {
            "candidate_id": self.candidate_id,
            "semantic_document": self.semantic_document,
            "lexical_document": self.lexical_document,
            "redrob_signals": self.redrob_signals,
        }


# ---------------------------------------------------------------------------
# 2. ALLOWED REDROB SIGNAL KEYS
# ---------------------------------------------------------------------------

# Only these keys are ever copied out of redrob_signals. Everything else
# (recruiter popularity, profile views, connections, salary, etc.) is
# dropped here at the source so it can never leak downstream.
_ALLOWED_REDROB_KEYS: tuple[str, ...] = (
    "profile_completeness_score",
    "open_to_work_flag",
    "last_active_date",
    "interview_completion_rate",
    "verified_email",
    "verified_phone",
)


def _extract_allowed_redrob_signals(raw_signals: dict[str, Any]) -> dict[str, Any]:
    """Copy out only the allow-listed redrob signal keys."""
    return {
        key: raw_signals.get(key)
        for key in _ALLOWED_REDROB_KEYS
        if key in raw_signals
    }


# ---------------------------------------------------------------------------
# 3. SEMANTIC DOCUMENT BUILDER
# ---------------------------------------------------------------------------

def _format_career_history_semantic(career_history: list[dict[str, Any]]) -> str:
    """
    Render career history as natural-language sentences for embedding.

    Each role becomes a short paragraph: title, duration context, and the
    free-text description (which is where most of the real signal lives —
    e.g. "built a recommendation system at a product company" per the JD's
    own guidance on what counts as a genuine fit).
    """
    if not career_history:
        return ""

    parts: list[str] = []
    for role in career_history:
        title = role.get("title", "").strip()
        company_context = "current role" if role.get("is_current") else "previous role"
        duration_months = role.get("duration_months", 0)
        description = role.get("description", "").strip()

        # Duration phrased naturally — embeddings respond better to natural
        # language than to raw numbers with no context.
        duration_phrase = (
            f"held for approximately {duration_months} months"
            if duration_months
            else ""
        )

        sentence = f"Worked as {title} ({company_context}"
        if duration_phrase:
            sentence += f", {duration_phrase}"
        sentence += "). "
        if description:
            sentence += description

        parts.append(sentence.strip())

    return " ".join(parts)


def _format_education_semantic(education: list[dict[str, Any]]) -> str:
    """Render education as a natural-language fragment."""
    if not education:
        return ""

    parts: list[str] = []
    for edu in education:
        degree = edu.get("degree", "").strip()
        field_of_study = edu.get("field_of_study", "").strip()
        if degree and field_of_study:
            parts.append(f"Holds a {degree} in {field_of_study}.")
        elif degree:
            parts.append(f"Holds a {degree}.")
        elif field_of_study:
            parts.append(f"Studied {field_of_study}.")

    return " ".join(parts)


def _format_skills_semantic(skills: list[dict[str, Any]]) -> str:
    """
    Render skills as a natural-language sentence rather than a raw list,
    since the semantic document is meant for embedding (not BM25).

    Proficiency is included because "advanced X" carries more semantic
    weight than a bare skill name.
    """
    if not skills:
        return ""

    skill_phrases: list[str] = []
    for skill in skills:
        name = skill.get("name", "").strip()
        proficiency = skill.get("proficiency", "").strip()
        if not name:
            continue
        if proficiency:
            skill_phrases.append(f"{proficiency} level {name}")
        else:
            skill_phrases.append(name)

    if not skill_phrases:
        return ""

    return "Technical skills include: " + ", ".join(skill_phrases) + "."


def build_semantic_document(candidate: dict[str, Any]) -> str:
    """
    Build the natural-language semantic document for E5 embedding.

    Combines, in order:
        1. Current title (sets the headline context)
        2. Professional summary (highest-signal free text)
        3. Career history (titles + descriptions)
        4. Education
        5. Technical skills

    Explicitly excludes: name, location, country, company size, salary,
    recruiter popularity, profile views, connections, endorsements.

    Parameters
    ----------
    candidate : dict
        One filtered candidate record (full schema from the gate stage).

    Returns
    -------
    str
        Natural-language document ready for sentence-transformers encoding.
    """
    profile: dict[str, Any] = candidate.get("profile", {})

    segments: list[str] = []

    # --- Current title ---
    current_title = profile.get("current_title", "").strip()
    if current_title:
        segments.append(f"Current role: {current_title}.")

    # --- Professional summary ---
    summary = profile.get("summary", "").strip()
    if summary:
        segments.append(summary)

    # --- Career history ---
    career_text = _format_career_history_semantic(candidate.get("career_history", []))
    if career_text:
        segments.append(career_text)

    # --- Education ---
    education_text = _format_education_semantic(candidate.get("education", []))
    if education_text:
        segments.append(education_text)

    # --- Skills ---
    skills_text = _format_skills_semantic(candidate.get("skills", []))
    if skills_text:
        segments.append(skills_text)

    # Join with single spaces, collapse any accidental double spaces.
    document = " ".join(segments)
    document = re.sub(r"\s+", " ", document).strip()

    return document


# ---------------------------------------------------------------------------
# 4. LEXICAL DOCUMENT BUILDER
# ---------------------------------------------------------------------------

# Stop-word-like filler that occasionally appears in proficiency strings or
# free text we pull tokens from — excluded so they don't pollute BM25 stats.
_LEXICAL_STOP_TOKENS: frozenset[str] = frozenset({
    "and", "or", "the", "a", "an", "of", "in", "for", "with", "to",
})


def _tokenize_for_lexical(text: str) -> list[str]:
    """
    Extract candidate technology/tool-like tokens from free text.

    Keeps alphanumeric tokens with internal hyphens, slashes, and dots
    (so "sentence-transformers", "a/b testing", "node.js" survive intact),
    lowercases everything, and drops short stop-word-like filler.
    """
    if not text:
        return []

    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+/\-.]*", text.lower())
    return [tok for tok in raw_tokens if tok not in _LEXICAL_STOP_TOKENS and len(tok) > 1]


def _extract_skill_tokens(skills: list[dict[str, Any]]) -> list[str]:
    """
    Pull raw skill names as lexical tokens, preserving multi-word skill
    names as a single hyphen-joined token where useful AND as separate
    words, so BM25 can match either "sentence transformers" style queries
    or "sentence-transformers" style queries.
    """
    tokens: list[str] = []
    for skill in skills:
        name = skill.get("name", "").strip()
        if not name:
            continue
        name_lower = name.lower()
        # Add the full skill name as a single token (spaces -> nothing lost,
        # BM25 treats this as a phrase-like atomic term when space-joined
        # naturally below; here we add the cleaned multi-word form).
        tokens.append(name_lower)
        # Also add individual words from multi-word skill names for partial
        # matching against single-word JD lexical terms.
        tokens.extend(_tokenize_for_lexical(name_lower))

    return tokens


def _extract_career_tech_tokens(career_history: list[dict[str, Any]]) -> list[str]:
    """
    Extract technology-flavoured tokens from career history titles and
    descriptions. We don't try to do full NLP entity extraction here —
    the lexical document is meant to be dense and inclusive, biasing
    toward recall (consistent with the gate's retrieval philosophy).
    """
    tokens: list[str] = []
    for role in career_history:
        title = role.get("title", "")
        description = role.get("description", "")
        tokens.extend(_tokenize_for_lexical(title))
        tokens.extend(_tokenize_for_lexical(description))
    return tokens


def build_lexical_document(candidate: dict[str, Any]) -> str:
    """
    Build the dense, non-natural-language lexical document for BM25.

    Composition (concatenated, space-separated, NOT prose):
        - All skill names (full + tokenized)
        - Technology-flavoured tokens extracted from career history
          titles and descriptions
        - Current title tokens

    This document intentionally maximizes lexical surface area so BM25
    has the best possible chance of matching JD technology terms, even
    when the candidate's summary doesn't explicitly name them.

    Explicitly excludes: name, location, country, company size, salary,
    recruiter popularity signals, education (low lexical value for tech
    matching; semantic document already covers it).

    Parameters
    ----------
    candidate : dict
        One filtered candidate record.

    Returns
    -------
    str
        Space-separated token string suitable for rank_bm25 tokenization
        (the BM25 module will .split() this string).
    """
    profile: dict[str, Any] = candidate.get("profile", {})

    tokens: list[str] = []

    # --- Current title tokens ---
    tokens.extend(_tokenize_for_lexical(profile.get("current_title", "")))

    # --- Skill tokens (full names + decomposed words) ---
    tokens.extend(_extract_skill_tokens(candidate.get("skills", [])))

    # --- Career history tech tokens ---
    tokens.extend(_extract_career_tech_tokens(candidate.get("career_history", [])))

    # Deduplicate while preserving order (stable for reproducibility), since
    # repeated tokens would artificially inflate BM25 term frequency for
    # things mentioned many times in free text vs. skills list.
    # NOTE: We deliberately do NOT deduplicate here — BM25's term-frequency
    # component is part of its scoring signal, and a tech mentioned in both
    # the skills list and the career description is a genuinely stronger
    # signal than one mentioned once. Deduplication would discard that.
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 5. TOP-LEVEL BUILDER — combines both documents for one candidate
# ---------------------------------------------------------------------------

def build_candidate_retrieval_document(
    candidate: dict[str, Any],
) -> CandidateRetrievalDocument:
    """
    Build the complete CandidateRetrievalDocument (semantic + lexical +
    allow-listed redrob signals) for a single candidate record.

    This is the single public entry point other modules should call.

    Parameters
    ----------
    candidate : dict
        One filtered candidate record (from filtered_candidates.jsonl).

    Returns
    -------
    CandidateRetrievalDocument
    """
    candidate_id: str = candidate.get("candidate_id", "UNKNOWN")

    semantic_document = build_semantic_document(candidate)
    lexical_document = build_lexical_document(candidate)

    raw_redrob_signals: dict[str, Any] = candidate.get("redrob_signals", {})
    allowed_redrob_signals = _extract_allowed_redrob_signals(raw_redrob_signals)

    return CandidateRetrievalDocument(
        candidate_id=candidate_id,
        semantic_document=semantic_document,
        lexical_document=lexical_document,
        redrob_signals=allowed_redrob_signals,
    )
