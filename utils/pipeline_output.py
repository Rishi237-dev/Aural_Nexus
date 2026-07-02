# ==========================================================
# utils/pipeline_output.py
#
# Responsibility: All logging, output file generation,
# runtime statistics, and diagnostics for the pipeline.
#
# This module has NO business logic and NO scoring logic.
# It only reads from the result dicts produced by
# calculate_final_score() and writes structured outputs.
#
# Public API
# ----------
# setup_logging()
#     Call once at pipeline startup. Configures the root
#     logger to write to both stdout and outputs/pipeline.log.
#
# write_all_outputs(results, candidates, start_time, data_path, skipped)
#     Convenience entry point. Calls all four writers in
#     order and logs completion of each.
#
# Individual writers (also callable directly):
#     write_diagnostics(...)    → outputs/diagnostics.json
#     write_run_summary(...)    → outputs/run_summary.json
#     write_ranked_csv(...)     → outputs/ranked_candidates.csv
# ==========================================================

import csv
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

OUTPUTS_DIR      = Path("outputs")
LOG_FILE         = OUTPUTS_DIR / "pipeline.log"
DIAGNOSTICS      = OUTPUTS_DIR / "diagnostics.json"
RUN_SUMMARY      = OUTPUTS_DIR / "run_summary.json"
RANKED_CSV       = OUTPUTS_DIR / "ranked_candidates.csv"

PIPELINE_VERSION = "1.0.0"

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """
    Configure the root logger once at pipeline startup.

    Attaches two handlers:
        StreamHandler  — writes INFO+ to stdout
        FileHandler    — writes INFO+ to outputs/pipeline.log
                         (overwrites on every run — always shows latest run only)

    Returns the root logger so callers can use logging.getLogger()
    conventionally without re-configuring.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging() is called twice
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — overwrite mode so each run starts with a fresh log
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger



# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def write_diagnostics(
    results: list,
    candidate_count: int,
    processed: int,
    skipped: int,
) -> None:
    """
    Write outputs/diagnostics.json.

    All metrics are derived strictly from the current
    calculate_final_score() return structure:
        result["final_score"]
        result["components"]    — retrieval, ranking, vector_db (all 0-100)
        result["vector"]        — consistency_score, credibility_multiplier,
                                  years_experience

    No legacy field names are referenced.

    Parameters
    ----------
    results          : list   Sorted list of calculate_final_score() dicts.
    candidate_count  : int    Total candidates loaded.
    processed        : int    Successfully scored candidates.
    skipped          : int    Candidates that raised an exception.
    """
    if not results:
        logging.warning("No results available — diagnostics file will be empty.")
        _write_json(DIAGNOSTICS, {})
        return

    scores            = [r["final_score"]                        for r in results]
    consistency_vals  = [r["vector"]["consistency_score"]        for r in results]
    credibility_vals  = [r["vector"]["credibility_multiplier"]   for r in results]
    experience_vals   = [r["vector"]["years_experience"]         for r in results]

    # Evidence coverage — derived from component scores (current API)
    # A component score > 0 means at least one keyword was found and
    # evidence exists for that dimension.
    with_retrieval  = sum(1 for r in results if r["components"]["retrieval"]  > 0)
    with_ranking    = sum(1 for r in results if r["components"]["ranking"]    > 0)
    with_vector_db  = sum(1 for r in results if r["components"]["vector_db"] > 0)

    payload = {
        "candidate_count":                   candidate_count,
        "processed_candidates":              processed,
        "skipped_candidates":                skipped,
        "highest_score":                     round(max(scores),              4),
        "lowest_score":                      round(min(scores),              4),
        "average_score":                     round(statistics.mean(scores),  4),
        "median_score":                      round(statistics.median(scores),4),
        "average_consistency":               round(statistics.mean(consistency_vals), 4),
        "average_credibility":               round(statistics.mean(credibility_vals), 4),
        "average_years_experience":          round(statistics.mean(experience_vals),  4),
        "candidates_with_retrieval_evidence": with_retrieval,
        "candidates_with_ranking_evidence":   with_ranking,
        "candidates_with_vector_db_evidence": with_vector_db,
    }

    _write_json(DIAGNOSTICS, payload)
    logging.info("Diagnostics written → %s", DIAGNOSTICS)


# ------------------------------------------------------------------
# Run summary
# ------------------------------------------------------------------

def write_run_summary(
    data_path: str,
    candidate_count: int,
    processed: int,
    skipped: int,
    total_runtime: float,
    results: list,
) -> None:
    """
    Write outputs/run_summary.json.

    A lightweight snapshot of the run — suitable for dashboards
    or CI checks without loading the full diagnostics file.

    Parameters
    ----------
    data_path       : str   Path to the input dataset.
    candidate_count : int   Total candidates loaded.
    processed       : int   Successfully scored candidates.
    skipped         : int   Candidates that raised an exception.
    total_runtime   : float Seconds from pipeline start to end.
    results         : list  Sorted list of calculate_final_score() dicts.
    """
    highest_score = round(results[0]["final_score"], 4) if results else 0.0
    avg_score = (
        round(statistics.mean(r["final_score"] for r in results), 4)
        if results
        else 0.0
    )

    payload = {
        "timestamp":         _utc_now(),
        "dataset":           data_path,
        "candidate_count":   candidate_count,
        "processed":         processed,
        "skipped":           skipped,
        "total_runtime_seconds": round(total_runtime, 4),
        "highest_score":     highest_score,
        "average_score":     avg_score,
        "pipeline_version":  PIPELINE_VERSION,
    }

    _write_json(RUN_SUMMARY, payload)
    logging.info("Run summary written → %s", RUN_SUMMARY)


# ------------------------------------------------------------------
# Ranked candidates CSV
# ------------------------------------------------------------------

def write_ranked_csv(results: list) -> None:
    """
    Write outputs/ranked_candidates.csv.

    Columns: rank, candidate_id, current_title, years_experience,
             final_score

    Results must already be sorted descending by final_score before
    this function is called. The rank column is generated here (1-based).

    All values come from the current calculate_final_score() return
    structure:
        result["candidate_id"]
        result["final_score"]
        result["vector"]["current_title"]
        result["vector"]["years_experience"]
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    with open(RANKED_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "candidate_id",
            "current_title",
            "years_experience",
            "final_score",
        ])

        for rank, result in enumerate(results, start=1):
            writer.writerow([
                rank,
                result["candidate_id"],
                result["vector"]["current_title"],
                result["vector"]["years_experience"],
                result["final_score"],
            ])

    logging.info(
        "Ranked candidates CSV written → %s  (%d rows)",
        RANKED_CSV,
        len(results),
    )


# ------------------------------------------------------------------
# Convenience entry point
# ------------------------------------------------------------------

def write_all_outputs(
    results: list,
    candidate_count: int,
    processed: int,
    skipped: int,
    start_time: float,
    data_path: str,
) -> None:
    """
    Write all three output files in one call.

    Called from main.py after the scoring loop and sort are complete.

    Parameters
    ----------
    results         : list   Sorted list of calculate_final_score() dicts.
    candidate_count : int    Total candidates loaded from dataset.
    processed       : int    Candidates scored without error.
    skipped         : int    Candidates skipped due to exceptions.
    start_time      : float  time.time() captured at pipeline start.
    data_path       : str    Path to the input dataset file.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    total_runtime = time.time() - start_time

    write_diagnostics(
        results=results,
        candidate_count=candidate_count,
        processed=processed,
        skipped=skipped,
    )

    write_run_summary(
        data_path=data_path,
        candidate_count=candidate_count,
        processed=processed,
        skipped=skipped,
        total_runtime=total_runtime,
        results=results,
    )

    write_ranked_csv(results=results)

    logging.info("Output generation completed.")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    """Serialise payload to JSON and write to path (UTF-8, 2-space indent)."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
