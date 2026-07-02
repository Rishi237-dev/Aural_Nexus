# ==========================================================
# utils/submission_generator.py
#
# Responsibility: Produce the official competition submission
# CSV from the full ranked candidate list.
#
# This module's output is graded by an external, byte-exact
# validator (validate_submission.py, provided by the
# organisers). The rules below are not stylistic choices —
# they are copied directly from submission_spec.md Section 2-3
# and validate_submission.py's REQUIRED_HEADER /
# CANDIDATE_ID_PATTERN / row-count / monotonicity checks.
#
# Hard requirements (auto-rejected by the validator otherwise):
#   - Filename: <participant_id>.csv  (caller's responsibility,
#     this module just writes to the given path)
#   - UTF-8 encoding
#   - Header row EXACTLY: candidate_id,rank,score,reasoning
#   - Exactly 100 data rows
#   - rank: integer 1-100, each used exactly once
#   - candidate_id: matches CAND_[0-9]{7}, no duplicates
#   - score: float, NON-INCREASING as rank increases
#   - ties in score: candidate_id must be ascending at the tie
#   - reasoning: non-empty (optional per spec, but penalised at
#     Stage 4 if empty — this module never writes an empty one)
#
# This module performs NO scoring and NO reasoning generation.
# It is a pure formatter over already-computed results.
# ==========================================================

import csv
import re
from pathlib import Path

from ranking.reasoning_generator import generate_reasoning


CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
SUBMISSION_ROW_COUNT = 100


def _validate_candidate_id(cid):
    """
    Defensive check before writing. If a candidate_id doesn't
    match the spec's pattern, it would be auto-rejected by the
    organiser's validator — better to catch and log it here
    than discover it after upload.
    """
    return bool(CANDIDATE_ID_PATTERN.match(cid))


def _enforce_tie_break(results):
    """
    Spec rule: if two candidates have the same final_score,
    ranks must still be assigned, and at equal scores the
    candidate_id must be ascending in rank order.

    `results` is already sorted by final_score descending
    (from signal_fusion / main.py). This function only
    reorders within tied-score groups — it never changes
    which candidates are present.
    """

    reordered = []
    i = 0
    n = len(results)

    while i < n:
        j = i
        # Find the full run of equal scores starting at i.
        while j + 1 < n and results[j + 1]["final_score"] == results[i]["final_score"]:
            j += 1

        tied_group = results[i:j + 1]

        if len(tied_group) > 1:
            tied_group = sorted(tied_group, key=lambda r: r["candidate_id"])

        reordered.extend(tied_group)
        i = j + 1

    return reordered


def generate_submission(
    ranked_results,
    output_path,
    logger=None,
):
    """
    Write the official top-100 submission CSV.

    Parameters
    ----------
    ranked_results : list[dict]
        Full list of calculate_final_score() outputs, already
        sorted by final_score descending. Must have length
        >= 100 (the full candidate pool, not pre-sliced) — this
        function performs the top-100 slice itself, after
        tie-breaking, so the slice boundary is correct even
        when scores tie exactly at rank 100.
    output_path : str or Path
        Where to write the CSV. Caller controls the filename
        (must be the participant's registered ID per spec
        Section 2 — this module does not enforce that part).
    logger : logging.Logger, optional
        If provided, submission generation events are logged
        through it, consistent with the rest of the pipeline's
        logging conventions. If None, this function is silent.

    Returns
    -------
    dict with keys:
        rows_written      int
        skipped_invalid_id int   — candidates dropped because
                                    their ID failed the spec
                                    pattern (logged, not fatal)
        output_path        str
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(
            f"Submission generation started — "
            f"{len(ranked_results)} candidates available."
        )

    # ----------------------------------------------------------
    # Step 1: tie-break the FULL list first.
    # Doing this before slicing matters: if rank 100 and rank
    # 101 are tied on score, the tie-break must be applied
    # before we cut, otherwise the wrong candidate could end up
    # inside the top 100 by accident of input order.
    # ----------------------------------------------------------

    ordered = _enforce_tie_break(ranked_results)

    # ----------------------------------------------------------
    # Step 2: filter invalid candidate_ids defensively, then
    # take the top 100 of what remains. This keeps the
    # submission valid even if an upstream parsing bug ever
    # produces a malformed ID.
    # ----------------------------------------------------------

    valid = []
    skipped_invalid_id = 0

    for r in ordered:
        if _validate_candidate_id(r["candidate_id"]):
            valid.append(r)
        else:
            skipped_invalid_id += 1
            if logger:
                logger.warning(
                    f"Skipped candidate with invalid ID format: "
                    f"{r['candidate_id']!r}"
                )

    if len(valid) < SUBMISSION_ROW_COUNT:
        if logger:
            logger.error(
                f"Only {len(valid)} valid candidates available — "
                f"fewer than the required {SUBMISSION_ROW_COUNT}. "
                f"Submission will be incomplete and will fail "
                f"validate_submission.py."
            )

    top_100 = valid[:SUBMISSION_ROW_COUNT]

    # ----------------------------------------------------------
    # Step 3: assign ranks 1-100 and generate reasoning.
    # Rank is the row's POSITION after tie-breaking — this is
    # the value written to the CSV, not anything stored
    # upstream, so it is always contiguous and correct.
    # ----------------------------------------------------------

    rows = []
    for position, result in enumerate(top_100, start=1):

        reasoning = generate_reasoning(result, rank=position)

        rows.append({
            "candidate_id": result["candidate_id"],
            "rank":         position,
            "score":        round(result["final_score"], 4),
            "reasoning":    reasoning,
        })

    # ----------------------------------------------------------
    # Step 4: monotonicity safety check.
    # Tie-breaking only reorders within equal-score groups, so
    # this should always hold by construction — but we assert
    # it explicitly rather than silently trust it, since a
    # validator rejection at upload time is far more costly
    # than a fast local check here.
    # ----------------------------------------------------------

    for k in range(len(rows) - 1):
        if rows[k]["score"] < rows[k + 1]["score"]:
            msg = (
                f"Monotonicity violation at rank "
                f"{rows[k]['rank']} -> {rows[k+1]['rank']}: "
                f"{rows[k]['score']} < {rows[k+1]['score']}"
            )
            if logger:
                logger.error(msg)
            raise ValueError(msg)

    # ----------------------------------------------------------
    # Step 5: write the CSV exactly to spec.
    # csv.writer with QUOTE_MINIMAL correctly quotes any
    # reasoning string containing commas — required since
    # reasoning text routinely contains commas.
    # ----------------------------------------------------------

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(REQUIRED_HEADER)

        for row in rows:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.4f}",
                row["reasoning"],
            ])

    if logger:
        logger.info(
            f"Submission written — {len(rows)} rows -> {output_path}"
        )
        if skipped_invalid_id:
            logger.warning(
                f"{skipped_invalid_id} candidates skipped due to "
                f"invalid ID format."
            )

    return {
        "rows_written":         len(rows),
        "skipped_invalid_id":   skipped_invalid_id,
        "output_path":          str(output_path),
    }
