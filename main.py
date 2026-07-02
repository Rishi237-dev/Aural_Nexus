"""
main.py — Pipeline orchestration entry point.

Pipeline order:
    JD Parser
    → Retrieval Pipeline  (E5 embeddings + FAISS + BM25 + RRF + Redrob)
    → Candidate Loader
    → JD Feature Extractor → Evidence Locator
    → Evidence Engine → Evidence Quality → Consistency Engine
    → Feature Vector → Signal Fusion → Submission Generation

No business logic lives here. This file only orchestrates
module calls and delegates all output/logging to
utils/pipeline_output.py.
"""

import json
import logging
import time
from pathlib import Path

from candidate.parser import load_candidates

from jd.jd_parser import generate_parsed_jd_file

from audit.dataset_audit import run_audit

from audit.jd_skill_audit import run_jd_skill_audit

from ranking.signal_fusion import calculate_final_score

from retrieval.run_retrieval import run_retrieval_pipeline

from utils.pipeline_output import setup_logging, write_all_outputs

from utils.submission_generator import generate_submission


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("data/candidates.jsonl")
LARGE_DATA_PATHS = [
    Path(r"D:\JBF_Test\data\candidates.jsonl"),
    Path(r"D:\Documents\JBF_Test\data\candidates.jsonl"),
]
DATA_PATH = next((str(path) for path in LARGE_DATA_PATHS if path.exists()), str(DEFAULT_DATA_PATH))
RUN_AUDIT    = False   # set True to run full dataset audit
RUN_JD_AUDIT = False   # set True to run JD skill frequency audit
TOP_N        = 20      # number of top-ranked candidates to print


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

setup_logging()

logging.info("Pipeline started.")


# ------------------------------------------------------------------
# Parse and write the canonical Job Description JSON.
# This must run before the ranking loop so that the structured
# JD is on disk and the jd_feature_extractor singleton is primed.
# ------------------------------------------------------------------

logging.info("Parsing Job Description → jd/parsed_job_description.json")
generate_parsed_jd_file(
    jd_path="jd/job_description.txt",
    output_path="jd/parsed_job_description.json",
)
logging.info("Job Description parsed and written.")


# ------------------------------------------------------------------
# Run retrieval pipeline.
# Generates retrieval/artifacts/retrieval_results.json using the
# parsed JD.  Existing embeddings and indexes are reused; only new
# candidates (if any) are re-embedded.
# ------------------------------------------------------------------

logging.info("Retrieval pipeline started.")
run_retrieval_pipeline(
    jd_path="jd/parsed_job_description.json",
)
logging.info("Retrieval pipeline completed.")


# ------------------------------------------------------------------
# Build retrieval score lookup from the pipeline output.
# The file is always fresh — written moments ago by run_retrieval_pipeline().
# ------------------------------------------------------------------

RETRIEVAL_RESULTS_PATH = "retrieval/artifacts/retrieval_results.json"

logging.info("Loading retrieval scores from %s", RETRIEVAL_RESULTS_PATH)

with open(RETRIEVAL_RESULTS_PATH, "r", encoding="utf-8") as _f:
    _retrieval_records = json.load(_f)

retrieval_score_lookup: dict[str, float] = {
    record["candidate_id"]: record["final_retrieval_score"]
    for record in _retrieval_records
}

_all_scores = list(retrieval_score_lookup.values())
retrieval_ext_min: float = min(_all_scores)
retrieval_ext_max: float = max(_all_scores)

logging.info(
    "Retrieval lookup built — %d entries  min=%.6f  max=%.6f.",
    len(retrieval_score_lookup),
    retrieval_ext_min,
    retrieval_ext_max,
)


# ------------------------------------------------------------------
# Load candidates
# ------------------------------------------------------------------

start = time.time()

candidates = load_candidates(DATA_PATH)

logging.info(
    "Dataset loaded: %s — %d candidates.",
    DATA_PATH,
    len(candidates),
)


# ------------------------------------------------------------------
# Optional: dataset audit
# ------------------------------------------------------------------

if RUN_AUDIT:
    logging.info("Dataset audit started.")
    run_audit(candidates)
    logging.info("Dataset audit completed.")


# ------------------------------------------------------------------
# Optional: JD skill audit
# ------------------------------------------------------------------

if RUN_JD_AUDIT:
    logging.info("JD skill audit started.")
    run_jd_skill_audit(candidates)
    logging.info("JD skill audit completed.")


# ------------------------------------------------------------------
# Pipeline: score every candidate
# ------------------------------------------------------------------

logging.info(
    "Ranking started — processing %d candidates.",
    len(candidates),
)

results  = []
skipped  = 0

for candidate in candidates:
    try:
        # Look up the pre-generated retrieval score; default to 0.0
        # for any candidate not present in the retrieval results.
        retrieval_score = retrieval_score_lookup.get(
            candidate.candidate_id, 0.0
        )
        result = calculate_final_score(
            candidate,
            retrieval_score,
            retrieval_ext_min,
            retrieval_ext_max,
        )
        results.append(result)
    except Exception as exc:
        skipped += 1
        logging.error(
            "Skipping candidate %s — %s: %s",
            getattr(candidate, "candidate_id", "<unknown>"),
            type(exc).__name__,
            exc,
        )

processed = len(results)

logging.info(
    "Ranking completed — processed: %d  skipped: %d.",
    processed,
    skipped,
)


# ------------------------------------------------------------------
# Sort by final_score descending
# ------------------------------------------------------------------

results.sort(key=lambda r: r["final_score"], reverse=True)


# ------------------------------------------------------------------
# Write all output files
# ------------------------------------------------------------------

write_all_outputs(
    results=results,
    candidate_count=len(candidates),
    processed=processed,
    skipped=skipped,
    start_time=start,
    data_path=DATA_PATH,
)

# ------------------------------------------------------------------
# Print diagnostic detail for the top candidate
# ------------------------------------------------------------------

if results:
    top = results[0]

    print("=" * 80)
    print("TOP CANDIDATE — FULL DIAGNOSTIC")
    print("=" * 80)

    print(f"\nCandidate ID : {top['candidate_id']}")
    print(f"Final Score  : {top['final_score']}")

    print("\nComponents:")
    for name, value in top["components"].items():
        weight       = top["weights"][name]
        contribution = round(value * weight / 100, 4)
        print(
            f"  {name:<12} score={value:>6.2f}"
            f"  weight={weight:>3}"
            f"  contribution={contribution:.4f}"
        )

    print("\nStrengths:")
    for s in top["strengths"]:
        print(f"  + {s}")

    print("\nWeaknesses:")
    for w in top["weaknesses"]:
        print(f"  - {w}")


# ------------------------------------------------------------------
# Print Top-N ranked candidates
# ------------------------------------------------------------------

print(f"\n{'=' * 80}")
print(f"TOP {TOP_N} RANKED CANDIDATES")
print(f"{'=' * 80}\n")

for rank, result in enumerate(results[:TOP_N], start=1):
    print(f"#{rank:<3} [{result['final_score']:>6.2f}]  {result['candidate_id']}")
    print(f"       Components : {result['components']}")
    print(f"       Strengths  : {result['strengths']}")
    print(f"       Weaknesses : {result['weaknesses']}")
    print()


# ------------------------------------------------------------------
# Generate official submission CSV (top 100)
# ------------------------------------------------------------------

submission_result = generate_submission(
    results,
    output_path="outputs/submission.csv",
    logger=logging.getLogger(),
)


# ------------------------------------------------------------------
# Final timing log
# ------------------------------------------------------------------

elapsed = time.time() - start

logging.info(
    "Pipeline finished — %.2fs  (%d candidates).",
    elapsed,
    len(candidates),
)