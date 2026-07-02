# ==========================================================
# jd/jd_parser.py
#
# Responsibility: Parse a raw Job Description text file into
# a fully structured JDSchema dataclass and serialise it to
# jd/parsed_job_description.json.
#
# Design rules:
#   1. This module is the ONLY place where the JD text is read
#      and interpreted. All other modules consume the parsed
#      representation — never the raw text directly.
#   2. All parsing is regex- and heuristic-based (stdlib only).
#      No ML or external dependencies are required.
#   3. The module is designed to be easily extended: adding a
#      new field only requires adding a new extraction function
#      and populating one extra field in JDSchema.
#   4. parse_job_description() is a pure function: same input
#      always produces the same output.
#   5. The canonical output path is jd/parsed_job_description.json.
# ==========================================================

import json
import re
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------
# Parser version — bump when extraction logic changes materially.
# ------------------------------------------------------------------

PARSER_VERSION = "1.1.0"

# ------------------------------------------------------------------
# Default source file path (relative to project root).
# ------------------------------------------------------------------

DEFAULT_JD_PATH   = "jd/job_description.txt"
DEFAULT_JSON_PATH = "jd/parsed_job_description.json"


# ==========================================================
# Section heading patterns
# Each entry is a tuple of (canonical_name, list_of_regex_patterns).
# Patterns are matched case-insensitively against stripped lines.
# ==========================================================

SECTION_PATTERNS = [
    ("responsibilities",   [
        r"^responsibilities",
        r"^what you.ll do",
        r"^key responsibilities",
        r"^the role",
        r"^your role",
        r"^what you will do",
        r"^what you.d actually be doing",
        r"^in practical terms",
    ]),
    ("requirements",       [
        r"^requirements",
        r"^required",
        r"^must have",
        r"^minimum qualifications",
        r"^what we.re looking for",
        r"^what you.ll need",
        r"^you have",
        r"^basic qualifications",
        r"^things you absolutely need",
        r"^the skills inventory",
    ]),
    ("preferred",          [
        r"^nice to have",
        r"^preferred",
        r"^bonus",
        r"^good to have",
        r"^preferred qualifications",
        r"^plus",
        r"^ideally",
        r"^things we.d like you to have",
    ]),
    ("behavioral",         [
        r"^we are looking for",
        r"^who you are",
        r"^about you",
        r"^soft skills",
        r"^you are",
        r"^interpersonal",
        r"^the vibe check",
        r"^how to read between the lines",
    ]),
    ("negative",           [
        r"^disqualifying",
        r"^not required",
        r"^will not",
        r"^dealbreaker",
        r"^exclusion",
        r"^things we explicitly do not want",
    ]),
    ("domain",             [
        r"^domain",
        r"^industry",
        r"^domain\s*/\s*industry",
    ]),
    ("about_company",      [
        r"^about us",
        r"^about the company",
        r"^who we are",
        r"^company\s+(?:overview|description|profile)",
        r"^let.s be honest about this role",
    ]),
    ("about_role",         [
        r"^about the role",
        r"^role overview",
        r"^overview",
        r"^the opportunity",
        r"^position overview",
        r"^what we mean by",
        r"^on location.+logistics",
        r"^final note",
    ]),
]


# ==========================================================
# Technology vocabulary
# Used to extract named technologies from free text.
# Grouped by subcategory for future filtering; the parser
# flattens them when populating required/preferred tech fields.
# ==========================================================

TECH_VOCABULARY = {
    "vector_db": [
        "faiss", "milvus", "qdrant", "pinecone", "weaviate",
        "pgvector", "chroma", "chromadb", "vespa", "marqo",
    ],
    "search_engines": [
        "elasticsearch", "opensearch", "solr", "typesense",
        "algolia", "sphinx",
    ],
    "embedding_models": [
        "sentence transformers", "sentence-transformers",
        "bge", "e5", "openai embeddings", "ada", "cohere embed",
        "instructor", "gte", "jina",
    ],
    "retrieval": [
        "bm25", "dense retrieval", "hybrid search",
        "semantic search", "vector search",
        "approximate nearest neighbour", "ann",
        "colbert", "splade", "dpr", "bi-encoder", "cross-encoder",
    ],
    "llm_techniques": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning",
        "rag", "retrieval-augmented generation",
        "query rewriting", "reranking", "re-ranking",
    ],
    "llm_models": [
        "llm", "llms", "gpt", "claude", "gemini", "llama",
        "mistral", "falcon", "phi",
    ],
    "languages": [
        "python", "java", "scala", "go", "rust", "c++", "c#",
    ],
    "ml_frameworks": [
        "pytorch", "tensorflow", "jax", "hugging face",
        "transformers", "scikit-learn",
    ],
    "evaluation": [
        "ndcg", "mrr", "map", "precision@k", "recall@k",
        "a/b testing", "ab testing",
    ],
    "ranking": [
        "learning to rank", "ltr", "lambdamart", "xgboost",
        "lightgbm", "ranker", "search ranking",
    ],
}

# Flat list for fast scanning — built once at module load time.
_ALL_TECH_TERMS = sorted(
    {term for terms in TECH_VOCABULARY.values() for term in terms},
    key=len,
    reverse=True,   # match longer terms first to avoid partial overlaps
)


# ==========================================================
# Evaluation metric patterns
# ==========================================================

EVALUATION_METRIC_PATTERNS = [
    r"\bndcg[@\s]?@?\s*\d*\b",
    r"\bmrr[@\s]?\d*\b",
    r"\bmap\b",
    r"\bprecision[@\s]?@\s*k\b",
    r"\brecall[@\s]?@\s*k\b",
    r"\ba/b\s+test(?:ing)?\b",
    r"\bab\s+test(?:ing)?\b",
]


# ==========================================================
# Domain keywords
# ==========================================================

DOMAIN_KEYWORDS = {
    "Search":                 ["search", "information retrieval", "web search"],
    "Information Retrieval":  ["information retrieval", "retrieval"],
    "NLP":                    ["nlp", "natural language", "text processing"],
    "Machine Learning":       ["machine learning", "ml engineering", "ml platform"],
    "AI Platform":            ["ai platform", "mlops", "model serving"],
    "Data Engineering":       ["data engineering", "data pipeline", "etl"],
    "Recommendation":         ["recommendation", "recommender"],
    "HR Tech":                ["hr-tech", "recruiting tech", "talent platform"],
    "AI Engineering":         ["ai engineer", "ai engineering", "ml systems"],
    "Talent Intelligence":    ["talent intelligence", "talent platform", "candidate discovery"],
    "Ranking Systems":        ["ranking system", "ranking systems", "learning to rank",
                               "search ranking", "re-ranking"],
}


# ==========================================================
# Concept keywords
# High-level domain concepts that should appear in the keyword
# index when they occur in the JD text.  These are distinct from
# TECH_VOCABULARY (which captures named tools/libraries) — concept
# keywords capture *what the role is about* at a semantic level.
# ==========================================================

CONCEPT_KEYWORDS = [
    "embeddings",
    "retrieval",
    "ranking",
    "candidate matching",
    "candidate-jd matching",
    "recruiter feedback",
    "recruiter engagement",
    "recruiter-engagement",
    "vector databases",
    "vector database",
    "evaluation framework",
    "evaluation infrastructure",
    "distributed systems",
    "inference optimization",
    "marketplace",
    "hybrid retrieval",
    "embedding drift",
    "index refresh",
    "offline benchmarks",
    "online a/b testing",
    "feedback loops",
    "mentoring",
]


# ==========================================================
# Employment type patterns
# ==========================================================

EMPLOYMENT_TYPE_PATTERNS = {
    "Full-time":  [r"full[\s-]?time"],
    "Part-time":  [r"part[\s-]?time"],
    "Contract":   [r"\bcontract\b", r"\bfreelance\b", r"\bcontractor\b"],
    "Remote":     [r"\bremote\b"],
    "Hybrid":     [r"\bhybrid\b"],
    "Internship": [r"\bintern(?:ship)?\b"],
}

# On-site is only emitted when the JD *explicitly* labels the role as
# on-site.  Incidental mentions like "in-office days" or "on-site
# collaboration" should NOT trigger it.  We check against a narrow
# set of preamble-style declarations rather than a broad text scan.
_ONSITE_EXPLICIT_PATTERNS = [
    # "On-site" / "Onsite" as a standalone employment-type label
    r"(?:employment|work)\s*(?:type|model|arrangement)\s*[:;\-]\s*(?:.*\b)?on[\s-]?site",
    r"\bon[\s-]?site\s+(?:role|position|opportunity)\b",
    r"\blocation\s*[:;\-]\s*(?:.*\b)?on[\s-]?site",
]


# ==========================================================
# JDSchema — the structured representation of a Job Description
# ==========================================================

@dataclass
class JDSchema:
    """
    Structured representation of a parsed Job Description.

    All list fields default to empty list so downstream code
    never needs to guard against None on list iteration.
    Optional[str] fields are None when the field could not
    be extracted from the text.
    """

    # ---- Identity ------------------------------------------------
    job_title:              Optional[str]       = None
    company:                Optional[str]       = None
    location:               Optional[str]       = None

    # ---- Classification ------------------------------------------
    employment_type:        list = field(default_factory=list)
    domain:                 list = field(default_factory=list)

    # ---- Experience ----------------------------------------------
    experience_requirements: dict = field(default_factory=dict)

    # ---- Skills & technologies -----------------------------------
    required_skills:        list = field(default_factory=list)
    preferred_skills:       list = field(default_factory=list)
    required_technologies:  list = field(default_factory=list)
    preferred_technologies: list = field(default_factory=list)

    # ---- Role content --------------------------------------------
    responsibilities:       list = field(default_factory=list)
    behavioral_expectations: list = field(default_factory=list)

    # ---- Evaluation & metrics ------------------------------------
    evaluation_metrics:     list = field(default_factory=list)

    # ---- Negative signals ----------------------------------------
    negative_requirements:  list = field(default_factory=list)

    # ---- Keyword index -------------------------------------------
    keywords:               list = field(default_factory=list)

    # ---- Provenance ----------------------------------------------
    raw_text:               str  = ""
    parse_metadata:         dict = field(default_factory=dict)


# ==========================================================
# Internal helpers
# ==========================================================

def _clean(text: str) -> str:
    """Strip leading bullets, dashes, and whitespace from a line."""
    return re.sub(r"^[\s\-\*\•\·\u2022\u2013\u2014]+", "", text).strip()


def _is_section_heading(line: str) -> Optional[str]:
    """
    Return the canonical section name if the stripped line matches
    any known section heading pattern, else return None.
    """
    stripped = line.strip().rstrip(":").rstrip("-").strip()
    for canonical, patterns in SECTION_PATTERNS:
        for pat in patterns:
            if re.match(pat, stripped, re.IGNORECASE):
                return canonical
    return None


def _split_into_sections(text: str) -> dict:
    """
    Split the raw JD text into a dict mapping section names to
    their content lines.

    Lines before any recognised heading go into "preamble".
    Lines under an unrecognised heading go into "misc".
    """
    sections: dict = {"preamble": []}
    current = "preamble"

    for line in text.splitlines():
        heading = _is_section_heading(line)
        if heading:
            current = heading
            if current not in sections:
                sections[current] = []
        else:
            stripped = line.strip()
            if stripped:
                sections.setdefault(current, []).append(stripped)

    return sections


def _extract_bullet_items(lines: list) -> list:
    """
    Return a deduplicated list of cleaned non-empty items from
    a section's lines.

    Two-pass strategy
    -----------------
    Pass 1 — join continuation lines:
        Many JDs word-wrap long bullet points across two physical lines.
        A line is treated as a continuation of the previous item when it:
          * Does not start with a bullet marker (-, *, •, digit + .)
          * Does not start with a parenthesised fragment like "(FAISS, …)"
            that is clearly a tag-line continuation.
        Continuation lines are appended to the previous item with a space.

    Pass 2 — filter and deduplicate:
        The following line types are dropped:
          - Separator lines made entirely of =, -, ~, *, # (heading underlines).
          - Short lines ending with ':' that look like sub-headings.
          - Exact-duplicate lines (case-insensitive).
    """
    # ------------------------------------------------------------------
    # Pass 1: join soft-wrapped continuation lines
    # ------------------------------------------------------------------
    _BULLET_START = re.compile(
        r"^(?:[-*\u2022\u2013\u2014]\s+|\d+[.)]\s+)"
    )

    joined = []
    for line in lines:
        cleaned = _clean(line)
        if not cleaned:
            continue
        # Drop separator-only lines immediately (no point joining them)
        if re.fullmatch(r"[=\-~*#]{2,}", cleaned):
            continue

        is_new_bullet = bool(_BULLET_START.match(line.strip()))

        if joined and not is_new_bullet:
            prev = joined[-1]
            # Merge if the previous item does not end with sentence-closing
            # punctuation — this covers both soft-wrapped continuations and
            # parenthesised tag-lines like "(FAISS, Qdrant, Milvus, …)."
            if not re.search(r"[.!?]\s*$", prev):
                joined[-1] = prev + " " + cleaned
                continue

        joined.append(cleaned)


    # ------------------------------------------------------------------
    # Pass 2: filter sub-headings and deduplicate
    # ------------------------------------------------------------------
    seen = set()
    items = []
    for item in joined:
        # Drop short sub-heading lines ending with ':'
        if item.endswith(":") and len(item) < 60:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            items.append(item)
    return items


def _filter_skill_items(items: list) -> list:
    """
    Remove non-skill descriptive sentences from a skills list.

    Many JDs (especially conversational ones) include preamble prose
    in the same section as actual requirements.  This filter drops
    lines that read as meta-commentary or narrative rather than a
    concrete skill, qualification, or experience requirement.

    Heuristics applied (all case-insensitive):
      1. Lines that don't mention any recognisable skill/tech signal
         AND read like editorial prose (no tech terms, no years-of-
         experience pattern, no tool/framework name).
      2. Lines explicitly flagged by tell-tale preamble phrases such
         as "most JDs", "we're going to", "this is the section",
         "please read", etc.
    """
    # Patterns that indicate meta-commentary, not a skill requirement
    _NON_SKILL_PATTERNS = [
        r"^most jds?\b",
        r"^we.re going to",
        r"^this is the section",
        r"^please read",
        r"^here.s what",
        r"^here are the",
        r"^we.ve listed",
        r"^the following",
        r"^in this section",
        r"^note that",
        r"^we expect",
        r"^we don.t expect",
    ]

    filtered = []
    for item in items:
        item_lower = item.lower().strip()
        is_noise = False
        for pat in _NON_SKILL_PATTERNS:
            if re.match(pat, item_lower):
                is_noise = True
                break
        if not is_noise:
            filtered.append(item)
    return filtered


def _find_tech_terms(text: str, vocab: list = None) -> list:
    """
    Scan text for technology terms from the vocabulary.
    Returns a deduplicated list preserving original casing from vocab.
    """
    if vocab is None:
        vocab = _ALL_TECH_TERMS
    text_lower = text.lower()
    found = []
    seen = set()
    for term in vocab:
        # Use word-boundary-aware matching where possible
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                # Return with proper casing from vocab
                found.append(term)
    return found


def _extract_job_title(preamble: list, raw_text: str) -> Optional[str]:
    """
    Try to extract the job title from the preamble or explicit labels.
    """
    label_patterns = [
        r"^(?:position|job\s+title|job\s+description|title|role)\s*[:\-]\s*(.+)",
    ]
    for line in preamble:
        for pat in label_patterns:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                return m.group(1).strip()

    # Fall back: first non-empty preamble line that isn't a label
    for line in preamble:
        cleaned = _clean(line)
        if cleaned and not re.match(
            r"^(company|location|employment|type|date|posted|about)",
            cleaned, re.IGNORECASE
        ):
            # Heuristic: job titles are typically < 80 chars and title-cased
            if len(cleaned) < 80:
                return cleaned

    return None


def _extract_company(preamble: list, about_company: list) -> Optional[str]:
    """Extract company name from explicit labels or 'About us' section."""
    label_pat = r"^(?:company|organisation|organization)\s*[:\-]\s*(.+)"
    for line in (preamble + about_company):
        m = re.match(label_pat, line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _extract_location(preamble: list, raw_text: str) -> Optional[str]:
    """Extract location from explicit labels or common patterns."""
    label_pat = r"^(?:location|based\s+in|office)\s*[:\-]\s*(.+)"
    for line in preamble:
        m = re.match(label_pat, line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Scan full text for location label
    m = re.search(
        r"(?:location|based\s+in)\s*[:\-]\s*([^\n]+)",
        raw_text, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    return None


def _extract_employment_type(raw_text: str) -> list:
    """
    Detect employment type tags from the full text.

    On-site is treated specially: it is only included when the JD
    contains an *explicit* on-site declaration (e.g. "Employment Type:
    On-site").  Incidental phrases like "in-office days" or "on-site
    collaboration" are ignored.
    """
    text_lower = raw_text.lower()
    found = []
    for label, patterns in EMPLOYMENT_TYPE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                found.append(label)
                break

    # On-site: only add when the JD explicitly declares it
    if "On-site" not in found:
        for pat in _ONSITE_EXPLICIT_PATTERNS:
            if re.search(pat, text_lower):
                found.append("On-site")
                break

    return found


def _extract_experience(raw_text: str) -> dict:
    """
    Extract experience requirements.

    Returns a dict with:
        min_years   : int or None
        max_years   : int or None
        raw         : list of raw matched strings
    """
    patterns = [
        r"(\d+)\+?\s*(?:[\-\u2013\u2014]\s*\d+)?\s*years?\s+(?:of\s+)?experience",
        r"minimum\s+(?:of\s+)?(\d+)\s*(?:to\s*\d+\s*)?years?",
        r"at\s+least\s+(\d+)\s*(?:to\s*\d+\s*)?years?",
        r"(\d+)\s*(?:to|[\-\u2013\u2014])\s*(\d+)\s*years?",
        r"experience\s+required\s*[:\-]\s*(\d+)\s*[\-\u2013\u2014]\s*(\d+)\s*years?",
    ]
    raw_matches = []
    min_years = None
    max_years = None

    for pat in patterns:
        for m in re.finditer(pat, raw_text, re.IGNORECASE):
            raw_matches.append(m.group(0).strip())
            val = int(m.group(1))
            if min_years is None or val < min_years:
                min_years = val
            if len(m.groups()) >= 2 and m.group(2):
                v2 = int(m.group(2))
                if max_years is None or v2 > max_years:
                    max_years = v2

    # Deduplicate raw matches
    seen = set()
    deduped = []
    for r in raw_matches:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return {
        "min_years": min_years,
        "max_years": max_years,
        "raw":       deduped,
    }


def _extract_evaluation_metrics(raw_text: str) -> list:
    """Extract evaluation metric mentions from the full text."""
    found = []
    seen = set()
    text_lower = raw_text.lower()
    for pat in EVALUATION_METRIC_PATTERNS:
        for m in re.finditer(pat, text_lower):
            metric = m.group(0).strip().upper()
            # Normalise spacing
            metric = re.sub(r"\s+", " ", metric)
            if metric not in seen:
                seen.add(metric)
                found.append(metric)
    return found


def _extract_domain(raw_text: str, domain_section: list) -> list:
    """
    Infer domain / industry from explicit section or keyword scan.
    """
    found = []
    # Prefer explicit domain section content
    domain_text = " ".join(domain_section).lower() if domain_section else ""
    scan_text   = domain_text if domain_text else raw_text.lower()

    for domain_label, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in scan_text:
                if domain_label not in found:
                    found.append(domain_label)
                break

    return found


def _find_concept_keywords(text: str) -> list:
    """
    Scan text for high-level concept keywords from CONCEPT_KEYWORDS.
    Returns a deduplicated list of matched concepts (lowercased).
    """
    text_lower = text.lower()
    found = []
    seen = set()
    for concept in CONCEPT_KEYWORDS:
        pattern = r"(?<![a-z0-9])" + re.escape(concept) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            # Normalise hyphenated variants to a single canonical form
            canonical = concept.replace("-", " ").replace("  ", " ").strip()
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
    return found


def _build_keyword_index(jd: "JDSchema") -> list:
    """
    Build a deduplicated, lowercase keyword index from all
    structured fields. This becomes the `keywords` field and
    is the primary input for the AI Embedding Generator.

    Sources:
      1. Named technologies (required + preferred)
      2. Evaluation metrics
      3. Domain tags and employment type
      4. Tech terms extracted from skill / responsibility /
         behavioral prose
      5. Concept keywords found anywhere in the raw JD text

    Full skill sentences from required_skills and preferred_skills
    are intentionally excluded because they would pollute the
    keyword index with prose fragments.
    """
    pool = set()

    def _add_list(lst):
        for item in lst:
            if isinstance(item, str):
                pool.add(item.lower().strip())

    # Technologies, metrics, and domain tags are already atomic terms.
    _add_list(jd.required_technologies)
    _add_list(jd.preferred_technologies)
    _add_list(jd.evaluation_metrics)
    _add_list(jd.domain)
    _add_list(jd.employment_type)

    if jd.job_title:
        pool.add(jd.job_title.lower().strip())

    # Extract tech terms from skill prose (required + preferred)
    skill_prose = " ".join(jd.required_skills + jd.preferred_skills)
    _add_list(_find_tech_terms(skill_prose))

    # Extract tech terms from responsibilities prose
    resp_text = " ".join(jd.responsibilities)
    _add_list(_find_tech_terms(resp_text))

    # Extract tech terms from behavioral prose
    behav_text = " ".join(jd.behavioral_expectations)
    _add_list(_find_tech_terms(behav_text))

    # Concept keywords from the full raw text
    _add_list(_find_concept_keywords(jd.raw_text))

    # ------------------------------------------------------------------
    # Deduplication: collapse near-duplicate variants.
    # e.g. "a/b testing" and "a/b test" -> keep the longer form.
    # e.g. "llm" and "llms" -> keep both (distinct meaning density).
    # ------------------------------------------------------------------
    normalised = set()
    for kw in pool:
        canon = kw.replace("-", " ").replace("  ", " ").strip()
        normalised.add(canon)

    return sorted(normalised)


# ==========================================================
# Public API
# ==========================================================

def parse_job_description(text: str, source_path: str = "") -> JDSchema:
    """
    Parse raw Job Description text into a structured JDSchema.

    Parameters
    ----------
    text : str
        The full raw text of the Job Description.
    source_path : str, optional
        Path to the source file, recorded in parse_metadata.

    Returns
    -------
    JDSchema
        A fully populated structured representation.
    """
    sections = _split_into_sections(text)

    preamble               = sections.get("preamble", [])
    responsibilities_lines = sections.get("responsibilities", [])
    requirements_lines     = sections.get("requirements", [])
    preferred_lines        = sections.get("preferred", [])
    # "behavioral" section: only lines that genuinely fell under a
    # recognised behavioral heading (e.g. "We Are Looking For").
    # The "about_role" section is excluded here so that intro copy
    # ("You will design, build …") does not pollute behavioral.
    about_role_lines       = sections.get("about_role", [])
    behavioral_raw_lines   = sections.get("behavioral", [])
    # Remove any intro lines shared between about_role and behavioral
    # (can occur when the role overview paragraph lands in the wrong bucket).
    about_role_set         = {l.strip().lower() for l in about_role_lines}
    behavioral_lines       = [
        l for l in behavioral_raw_lines
        if l.strip().lower() not in about_role_set
    ]
    negative_lines         = sections.get("negative", [])
    domain_lines           = sections.get("domain", [])
    about_company_lines    = sections.get("about_company", [])

    # Full text for cross-section regex scans
    requirements_text = "\n".join(requirements_lines)
    preferred_text    = "\n".join(preferred_lines)
    resp_text         = "\n".join(responsibilities_lines)

    # ---- Build JDSchema ------------------------------------------

    jd = JDSchema(raw_text=text)

    # Identity
    jd.job_title = _extract_job_title(preamble, text)
    jd.company   = _extract_company(preamble, about_company_lines)
    jd.location  = _extract_location(preamble, text)

    # Classification
    jd.employment_type = _extract_employment_type(text)
    jd.domain          = _extract_domain(text, domain_lines)

    # Experience
    jd.experience_requirements = _extract_experience(text)

    # Skills: bullet items from the requirements / preferred sections.
    # These are full prose lines (e.g. "5+ years of experience in …").
    # Tech term extraction from these lines feeds required_technologies.
    jd.required_skills  = _filter_skill_items(
        _extract_bullet_items(requirements_lines)
    )
    jd.preferred_skills = _extract_bullet_items(preferred_lines)

    # Technologies: named entity scan across requirement text and
    # responsibility descriptions. Preferred technologies are scanned
    # separately from the nice-to-have section only.
    jd.required_technologies  = _find_tech_terms(requirements_text + " " + resp_text)
    jd.preferred_technologies = _find_tech_terms(preferred_text)

    # Responsibilities: bullet items from the responsibilities section.
    jd.responsibilities = _extract_bullet_items(responsibilities_lines)

    # Behavioral expectations: only from the dedicated behavioral
    # section (e.g. "We Are Looking For"), not from the role intro.
    jd.behavioral_expectations = _extract_bullet_items(behavioral_lines)

    # Evaluation metrics
    jd.evaluation_metrics = _extract_evaluation_metrics(text)

    # Negative / disqualifying requirements
    jd.negative_requirements = _extract_bullet_items(negative_lines)

    # Keyword index (built last, depends on all other fields)
    jd.keywords = _build_keyword_index(jd)

    # Provenance
    jd.parse_metadata = {
        "parser_version": PARSER_VERSION,
        "parsed_at":      datetime.now(timezone.utc).isoformat(),
        "source_file":    source_path or DEFAULT_JD_PATH,
    }

    return jd


def jd_to_dict(jd: JDSchema) -> dict:
    """
    Serialise a JDSchema to a plain dict suitable for JSON output.
    The raw_text field is included for full fidelity.
    """
    return asdict(jd)


def load_and_parse_jd(path: str = DEFAULT_JD_PATH) -> JDSchema:
    """
    Read a Job Description text file and return a parsed JDSchema.

    Parameters
    ----------
    path : str
        Path to the .txt file (relative to the project root).

    Returns
    -------
    JDSchema
    """
    resolved = Path(path)
    with open(resolved, "r", encoding="utf-8") as fh:
        text = fh.read()
    return parse_job_description(text, source_path=str(resolved))


def generate_parsed_jd_file(
    jd_path:    str = DEFAULT_JD_PATH,
    output_path: str = DEFAULT_JSON_PATH,
) -> JDSchema:
    """
    Load, parse, and write the structured JD to a JSON file.

    This is the primary entry-point called from main.py.
    It loads the raw JD text, runs the full parser, writes the
    canonical JSON output, and returns the JDSchema for in-process
    use (e.g. by jd_feature_extractor).

    Parameters
    ----------
    jd_path : str
        Path to the raw JD text file.
    output_path : str
        Destination path for the JSON output.

    Returns
    -------
    JDSchema
        The parsed schema (also written to output_path).
    """
    jd = load_and_parse_jd(jd_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(jd_to_dict(jd), fh, indent=2, ensure_ascii=False)

    return jd


# ==========================================================
# CLI entry-point
# ==========================================================

if __name__ == "__main__":
    jd = generate_parsed_jd_file()
    print(f"Parsed JD written to {DEFAULT_JSON_PATH}")
    print(f"  Title   : {jd.job_title}")
    print(f"  Company : {jd.company}")
    print(f"  Location: {jd.location}")
