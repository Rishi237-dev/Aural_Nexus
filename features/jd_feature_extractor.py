# ==========================================================
# features/jd_feature_extractor.py
#
# Responsibility: Match candidate profile text against the
# structured Job Description and return keyword-match counts.
#
# This module bridges the structured JD representation
# (jd/jd_parser.py) and the existing pipeline. It has two
# distinct concerns:
#
#   1. JD parsing — exposed via get_parsed_jd() and the
#      generate_parsed_jd_file() re-export. The JDSchema is
#      loaded ONCE per process run and cached.
#
#   2. Candidate matching — extract_jd_features(candidate)
#      scans the candidate's profile text for JD-derived
#      keywords and returns the same flat dict shape that
#      feature_vector.py has always expected. This preserves
#      full backward compatibility with every downstream module.
#
# Backward-compatibility contract:
#   extract_jd_features(candidate) returns exactly:
#     {
#       "<category>_score":   int   (count of matched keywords)
#       "<category>_matches": list  (the matched keywords)
#     }
#   for the seven categories: retrieval, vector_db, embeddings,
#   ranking, evaluation, llm, python.
#
# Extension points:
#   - To add a new JD category: add it to _build_match_categories()
#     and surface its score/matches keys in the return dict.
#   - To change which keywords define a category: update
#     jd/job_description.txt and/or TECH_VOCABULARY in jd_parser.py.
#   - The structured JD (JDSchema) is available via get_parsed_jd()
#     for any future AI layer that needs the full representation.
# ==========================================================

from jd.jd_parser import (
    JDSchema,
    generate_parsed_jd_file,
    load_and_parse_jd,
)


# ------------------------------------------------------------------
# JD path constants — single place to change if the file moves.
# ------------------------------------------------------------------

JD_TEXT_PATH = "jd/job_description.txt"
JD_JSON_PATH = "jd/parsed_job_description.json"


# ------------------------------------------------------------------
# Module-level singleton cache.
# The JD is loaded and parsed exactly ONCE per process invocation.
# All calls to get_parsed_jd() after the first return the cached
# JDSchema without re-reading or re-parsing the file.
# ------------------------------------------------------------------

_parsed_jd: JDSchema | None = None


def get_parsed_jd() -> JDSchema:
    """
    Return the cached JDSchema, loading it on first call.

    Returns
    -------
    JDSchema
        The fully parsed Job Description for this process run.
    """
    global _parsed_jd
    if _parsed_jd is None:
        _parsed_jd = load_and_parse_jd(JD_TEXT_PATH)
    return _parsed_jd


# ------------------------------------------------------------------
# Category keyword map.
# Built ONCE from the parsed JD and cached at module level.
# Maps the seven legacy category names to keyword lists drawn
# from the structured JD representation.
# ------------------------------------------------------------------

def _build_match_categories(jd: JDSchema) -> dict:
    """
    Derive the seven match-category keyword lists from the JDSchema.

    Each category maps to a curated set of keywords drawn from:
      - jd.required_technologies
      - jd.preferred_technologies
      - jd.evaluation_metrics (lowercased)
      - jd.required_skills / preferred_skills subsets
      - Hard-coded domain-specific expansions that remain stable
        regardless of what the JD text says (e.g. "retrieval" will
        always be in the retrieval category).

    This means the keyword coverage is driven by the JD content
    rather than a manually maintained constant dict.
    """

    req_tech  = {t.lower() for t in jd.required_technologies}
    pref_tech = {t.lower() for t in jd.preferred_technologies}
    all_tech  = req_tech | pref_tech
    metrics   = {m.lower() for m in jd.evaluation_metrics}

    # ---- retrieval -----------------------------------------------
    retrieval_kw = [
        "retrieval", "information retrieval", "semantic search",
        "dense retrieval", "hybrid search", "bm25", "vector search",
        "approximate nearest neighbour", "ann", "dpr",
    ]
    retrieval_kw += [t for t in all_tech if any(
        kw in t for kw in ("retrieval", "search", "bm25", "ann", "dpr", "splade")
    )]

    # ---- vector_db -----------------------------------------------
    vector_db_kw = [
        "milvus", "qdrant", "pinecone", "weaviate", "faiss",
        "pgvector", "elasticsearch", "opensearch", "chroma",
        "chromadb", "vespa", "marqo",
    ]
    vector_db_kw += [t for t in all_tech if any(
        kw in t for kw in ("milvus", "qdrant", "pinecone", "weaviate",
                           "faiss", "pgvector", "elasticsearch",
                           "opensearch", "chroma", "vespa")
    )]

    # ---- embeddings ----------------------------------------------
    embeddings_kw = [
        "embedding", "embeddings", "sentence transformers",
        "sentence-transformers", "bge", "e5", "openai embeddings",
        "instructor", "gte", "jina", "cohere embed",
    ]
    embeddings_kw += [t for t in all_tech if any(
        kw in t for kw in ("embed", "bge", "e5", "instructor",
                           "sentence-transform", "sentence transform")
    )]

    # ---- ranking -------------------------------------------------
    ranking_kw = [
        "ranking", "learning to rank", "search ranking", "ranker",
        "ltr", "lambdamart", "colbert", "bi-encoder", "cross-encoder",
        "reranking", "re-ranking",
    ]
    ranking_kw += [t for t in all_tech if any(
        kw in t for kw in ("rank", "ltr", "lambdamart",
                           "colbert", "cross-encoder", "bi-encoder")
    )]

    # ---- evaluation ----------------------------------------------
    evaluation_kw = [
        "ndcg", "mrr", "map", "a/b testing", "precision@k",
        "recall@k", "ab testing",
    ]
    evaluation_kw += [m for m in metrics]

    # ---- llm -----------------------------------------------------
    llm_kw = [
        "llm", "llms", "lora", "qlora", "peft", "fine-tuning",
        "fine tuning", "rag", "retrieval-augmented generation",
        "query rewriting", "gpt", "claude", "gemini", "llama",
        "mistral",
    ]
    llm_kw += [t for t in all_tech if any(
        kw in t for kw in ("lora", "qlora", "peft", "fine-tun",
                           "rag", "llm", "gpt", "claude",
                           "gemini", "llama", "mistral")
    )]

    # ---- python --------------------------------------------------
    python_kw = ["python"]

    def _dedup(lst):
        seen = set()
        out = []
        for item in lst:
            k = item.lower()
            if k not in seen:
                seen.add(k)
                out.append(item.lower())
        return out

    return {
        "retrieval":  _dedup(retrieval_kw),
        "vector_db":  _dedup(vector_db_kw),
        "embeddings": _dedup(embeddings_kw),
        "ranking":    _dedup(ranking_kw),
        "evaluation": _dedup(evaluation_kw),
        "llm":        _dedup(llm_kw),
        "python":     _dedup(python_kw),
    }


# Cache the category map so it is only built once.
_match_categories: dict | None = None


def _get_match_categories() -> dict:
    """Return the cached category → keyword-list map."""
    global _match_categories
    if _match_categories is None:
        _match_categories = _build_match_categories(get_parsed_jd())
    return _match_categories


# ==========================================================
# Public API — backward-compatible pipeline entry-point
# ==========================================================

def extract_jd_features(candidate) -> dict:
    """
    Match the candidate's full profile text against JD-derived
    keyword categories.

    This function retains the exact return shape consumed by
    feature_vector.py. The keyword lists are now driven by the
    structured JDSchema instead of a hardcoded constant dict,
    but the scoring logic and output keys are identical to the
    previous implementation.

    Parameters
    ----------
    candidate : Candidate
        The candidate dataclass produced by candidate/parser.py.

    Returns
    -------
    dict
        Keys: "<category>_score" (int) and "<category>_matches" (list)
        for each of: retrieval, vector_db, embeddings, ranking,
        evaluation, llm, python.
    """

    # Build the candidate's full text corpus (same sections as before)
    text_parts = [
        candidate.headline,
        candidate.summary,
    ]
    text_parts.extend(s["name"] for s in candidate.skills)
    for job in candidate.career_history:
        text_parts.append(job.get("description", ""))

    full_text = " ".join(text_parts).lower()

    categories = _get_match_categories()
    features   = {}

    for category_name, keywords in categories.items():
        matches = [kw for kw in keywords if kw in full_text]
        features[f"{category_name}_score"]   = len(matches)
        features[f"{category_name}_matches"] = matches

    return features