from collections import Counter
from pathlib import Path

from tqdm import tqdm


JD_TERMS = [
    "retrieval",
    "ranking",
    "embeddings",
    "embedding",
    "vector search",
    "semantic search",
    "hybrid search",
    "bm25",

    "milvus",
    "qdrant",
    "pinecone",
    "weaviate",
    "faiss",
    "pgvector",

    "sentence transformers",
    "bge",
    "e5",

    "ndcg",
    "mrr",
    "map",
    "a/b testing",

    "lora",
    "qlora",
    "peft",

    "fine-tuning",
    "fine tuning",

    "llm",
    "llms",

    "python"
]


def run_jd_skill_audit(candidates):

    counts = Counter()

    for candidate in tqdm(candidates):

        text_parts = []

        text_parts.append(
            candidate.headline
        )

        text_parts.append(
            candidate.summary
        )

        text_parts.extend(
            s["name"] for s in candidate.skills
        )

        for job in candidate.career_history:

            text_parts.append(
                job.get(
                    "description",
                    ""
                )
            )

        full_text = " ".join(
            text_parts
        ).lower()

        for term in JD_TERMS:

            if term.lower() in full_text:

                counts[term] += 1

    report = []

    report.append("=" * 60)
    report.append("JD SKILL AUDIT")
    report.append("=" * 60)

    for term, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        pct = (
            count /
            len(candidates)
        ) * 100

        report.append(
            f"{term:<25} "
            f"{count:>8} "
            f"({pct:.2f}%)"
        )

    report_text = "\n".join(
        report
    )

    Path("outputs").mkdir(
        exist_ok=True
    )

    with open(
        "outputs/jd_skill_audit.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    print(
        "\nJD Skill Audit Complete"
    )

    print(
        "Saved to outputs/jd_skill_audit.txt"
    )