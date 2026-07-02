"""
jd_document_builder.py
========================
Builds the two retrieval query documents from the parsed Job Description JSON:

    1. Semantic JD Document — natural-language text for E5 query embedding.
    2. Lexical JD Document  — dense token list for BM25 querying.

This mirrors candidate_document_builder.py so that the semantic document and
lexical document live in the same representation space as the candidate
documents they will be compared against (same vocabulary style, same level
of "naturalness" vs "denseness").

Per the architecture spec: do NOT concatenate raw JSON fields verbatim.
Each field is rendered into meaningful structured text/tokens, mirroring
how a human would actually read and internalize the JD.

This module performs NO embedding, NO retrieval, NO scoring — it only
transforms the parsed JD dict into two plain text strings.

CPU-only. No external APIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# 1. OUTPUT DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class JDRetrievalDocument:
    """
    Container for the JD's two retrieval/query documents.

    Attributes
    ----------
    semantic_document : str
        Natural-language text for E5 query embedding.
    lexical_document : str
        Dense space-separated token list for BM25 querying.
    """
    semantic_document: str
    lexical_document: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dict."""
        return {
            "semantic_document": self.semantic_document,
            "lexical_document": self.lexical_document,
        }


# ---------------------------------------------------------------------------
# 2. SEMANTIC JD DOCUMENT BUILDER
# ---------------------------------------------------------------------------

def _format_experience_semantic(experience_requirements: dict[str, Any]) -> str:
    """Render the experience band as a natural-language sentence."""
    min_years = experience_requirements.get("min_years")
    max_years = experience_requirements.get("max_years")

    if min_years is None and max_years is None:
        return ""

    if min_years is not None and max_years is not None:
        return (
            f"This role is looking for a candidate with approximately "
            f"{min_years} to {max_years} years of relevant professional experience."
        )
    if min_years is not None:
        return f"This role requires at least {min_years} years of relevant experience."
    return f"This role is suitable for candidates with up to {max_years} years of experience."


def _format_skills_semantic(skills: list[str], label: str) -> str:
    """
    Render a list of verbose skill-requirement sentences (as they already
    appear in the parsed JD) into a clearly labeled semantic paragraph.

    The parsed JD's required_skills / preferred_skills entries are already
    natural-language sentences, so we mainly need to join them with a
    framing sentence rather than re-paraphrase them.
    """
    if not skills:
        return ""

    framing = f"{label}: "
    return framing + " ".join(s.strip() for s in skills if s.strip())


def _format_responsibilities_semantic(responsibilities: list[str]) -> str:
    """Render the responsibilities list as a natural-language paragraph."""
    if not responsibilities:
        return ""

    return "Key responsibilities for this role include: " + " ".join(
        r.strip() for r in responsibilities if r.strip()
    )


def _format_domain_semantic(domain: list[str]) -> str:
    """Render the domain tags as a natural-language sentence."""
    if not domain:
        return ""

    return f"This role sits within the following domains: {', '.join(domain)}."


def _format_negative_requirements_semantic(negative_requirements: list[str]) -> str:
    """
    Render negative requirements as a natural-language paragraph.

    These ARE included in the semantic document (unlike candidate documents,
    which have no equivalent concept) because the embedding model benefits
    from understanding what profile the JD is explicitly NOT looking for —
    this shapes the query vector away from those archetypes. The actual
    rejection logic for negative requirements lives in the upstream Candidate
    Gate; here we are only enriching the query's semantic context.
    """
    if not negative_requirements:
        return ""

    # Skip the first entry if it's just a section header (as in this JD's
    # parsed output: "This is the section most JDs skip...").
    meaningful = [
        n.strip() for n in negative_requirements
        if n.strip() and not n.strip().lower().startswith("this is the section")
    ]
    if not meaningful:
        return ""

    return "This role explicitly does not fit candidates who: " + " ".join(meaningful)


def build_jd_semantic_document(jd: dict[str, Any]) -> str:
    """
    Build the natural-language semantic document for E5 query embedding.

    Combines, in order:
        1. Job title
        2. Domain tags
        3. Experience band
        4. Required skills (verbose sentences, already natural language)
        5. Preferred skills
        6. Responsibilities
        7. Negative requirements (semantic context only)

    Explicitly excludes: company name, location/logistics, raw_text blob,
    parse_metadata, keywords array (that's lexical, not semantic).

    Parameters
    ----------
    jd : dict
        The parsed JD JSON.

    Returns
    -------
    str
        Natural-language document ready for sentence-transformers encoding.
    """
    segments: list[str] = []

    # --- Job title ---
    job_title = jd.get("job_title", "").strip()
    if job_title:
        segments.append(f"Job title: {job_title}.")

    # --- Domain ---
    domain_text = _format_domain_semantic(jd.get("domain", []))
    if domain_text:
        segments.append(domain_text)

    # --- Experience ---
    experience_text = _format_experience_semantic(jd.get("experience_requirements", {}))
    if experience_text:
        segments.append(experience_text)

    # --- Required skills ---
    required_text = _format_skills_semantic(
        jd.get("required_skills", []), "Required skills and experience"
    )
    if required_text:
        segments.append(required_text)

    # --- Preferred skills ---
    preferred_text = _format_skills_semantic(
        jd.get("preferred_skills", []), "Preferred (nice-to-have) skills"
    )
    if preferred_text:
        segments.append(preferred_text)

    # --- Responsibilities ---
    responsibilities_text = _format_responsibilities_semantic(jd.get("responsibilities", []))
    if responsibilities_text:
        segments.append(responsibilities_text)

    # --- Negative requirements (semantic framing only) ---
    negative_text = _format_negative_requirements_semantic(jd.get("negative_requirements", []))
    if negative_text:
        segments.append(negative_text)

    document = " ".join(segments)
    document = re.sub(r"\s+", " ", document).strip()

    return document


# ---------------------------------------------------------------------------
# 3. LEXICAL JD DOCUMENT BUILDER
# ---------------------------------------------------------------------------

_LEXICAL_STOP_TOKENS: frozenset[str] = frozenset({
    "and", "or", "the", "a", "an", "of", "in", "for", "with", "to",
})


def _tokenize_for_lexical(text: str) -> list[str]:
    """
    Extract technology/tool-like tokens from free text.
    Mirrors the tokenizer in candidate_document_builder.py so both sides
    of the BM25 comparison use an identical tokenization scheme.
    """
    if not text:
        return []

    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+/\-.]*", text.lower())
    return [tok for tok in raw_tokens if tok not in _LEXICAL_STOP_TOKENS and len(tok) > 1]


def build_jd_lexical_document(jd: dict[str, Any]) -> str:
    """
    Build the dense, non-natural-language lexical document for BM25 querying.

    Composition (concatenated, space-separated, NOT prose):
        - required_technologies (highest priority — repeated for term-frequency weight)
        - preferred_technologies
        - keywords (the JD's own curated keyword list)
        - tokens extracted from required_skills / preferred_skills sentences
          (catches technology mentions embedded in prose, e.g. "FAISS" inside
          a sentence rather than the standalone technologies list)

    Required technologies are intentionally repeated (added twice) relative
    to preferred technologies, since BM25 term frequency directly affects
    score, and the JD itself signals required >> preferred in importance.

    Parameters
    ----------
    jd : dict
        The parsed JD JSON.

    Returns
    -------
    str
        Space-separated token string for rank_bm25 querying.
    """
    tokens: list[str] = []

    # --- Required technologies (weighted via repetition) ---
    required_technologies: list[str] = jd.get("required_technologies", [])
    for tech in required_technologies:
        tech_clean = tech.strip().lower()
        if tech_clean:
            tokens.append(tech_clean)
            tokens.append(tech_clean)  # repeated: required carries more weight

    # --- Preferred technologies ---
    preferred_technologies: list[str] = jd.get("preferred_technologies", [])
    for tech in preferred_technologies:
        tech_clean = tech.strip().lower()
        if tech_clean:
            tokens.append(tech_clean)

    # --- Curated keywords list ---
    keywords: list[str] = jd.get("keywords", [])
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if kw_clean:
            tokens.append(kw_clean)

    # --- Tokens extracted from required/preferred skill sentences ---
    for sentence in jd.get("required_skills", []):
        tokens.extend(_tokenize_for_lexical(sentence))
    for sentence in jd.get("preferred_skills", []):
        tokens.extend(_tokenize_for_lexical(sentence))

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 4. TOP-LEVEL BUILDER — combines both documents for the JD
# ---------------------------------------------------------------------------

def build_jd_retrieval_document(jd: dict[str, Any]) -> JDRetrievalDocument:
    """
    Build the complete JDRetrievalDocument (semantic + lexical) from the
    parsed JD JSON.

    This is the single public entry point other modules should call.
    Called exactly once per retrieval run (the JD does not change per
    candidate).

    Parameters
    ----------
    jd : dict
        The parsed JD JSON (as loaded from parsed_job_description.json).

    Returns
    -------
    JDRetrievalDocument
    """
    semantic_document = build_jd_semantic_document(jd)
    lexical_document = build_jd_lexical_document(jd)

    return JDRetrievalDocument(
        semantic_document=semantic_document,
        lexical_document=lexical_document,
    )