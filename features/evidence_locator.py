from collections import defaultdict


EVIDENCE_CATEGORIES = {

    "retrieval": [
        "retrieval",
        "information retrieval",
        "semantic search",
        "dense retrieval",
        "hybrid search",
        "bm25",
        "vector search"
    ],

    "vector_db": [
        "milvus",
        "qdrant",
        "pinecone",
        "weaviate",
        "faiss",
        "pgvector",
        "elasticsearch",
        "opensearch"
    ],

    "embeddings": [
        "embedding",
        "embeddings",
        "sentence transformers",
        "bge",
        "e5"
    ],

    "ranking": [
        "ranking",
        "learning to rank",
        "search ranking",
        "ranker"
    ],

    "evaluation": [
        "ndcg",
        "mrr",
        "map",
        "a/b testing"
    ],

    "llm": [
        "llm",
        "llms",
        "lora",
        "qlora",
        "peft",
        "fine-tuning",
        "fine tuning"
    ],

    "python": [
        "python"
    ]
}


def _find_matches(text, keywords):

    matches = []

    text = text.lower()

    for keyword in keywords:

        if keyword.lower() in text:

            matches.append(keyword)

    return matches


def locate_evidence(candidate):

    evidence = {}

    headline = candidate.headline.lower()
    summary = candidate.summary.lower()

    skills_text = " ".join(
        s["name"] for s in candidate.skills
    ).lower()

    career_text = " ".join(
        job.get(
            "description",
            ""
        )
        for job in candidate.career_history
    ).lower()

    for category, keywords in (
        EVIDENCE_CATEGORIES.items()
    ):

        evidence[category] = {

            "headline":
                _find_matches(
                    headline,
                    keywords
                ),

            "summary":
                _find_matches(
                    summary,
                    keywords
                ),

            "skills":
                _find_matches(
                    skills_text,
                    keywords
                ),

            "career":
                _find_matches(
                    career_text,
                    keywords
                )
        }

    return evidence


def count_evidence_sources(
    category_evidence
):

    source_count = 0

    for source in (
        "headline",
        "summary",
        "skills",
        "career"
    ):

        if category_evidence[source]:

            source_count += 1

    return source_count


def count_total_matches(
    category_evidence
):

    total = 0

    for matches in (
        category_evidence.values()
    ):

        total += len(matches)

    return total