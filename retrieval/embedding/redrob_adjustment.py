"""
redrob_adjustment.py
======================
Computes a Redrob confidence multiplier from objective availability
signals, applied to the hybrid retrieval score:

    final_retrieval_score = hybrid_retrieval_score × redrob_multiplier

Per architecture spec, ONLY these signals may be used:
    - profile_completeness_score
    - open_to_work_flag
    - last_active_date
    - interview_completion_rate
    - verified_email
    - verified_phone

Explicitly forbidden (these belong to the later ranking stage, not
retrieval):
    - recruiter popularity (saved_by_recruiters_30d, search_appearance_30d,
      recruiter_response_rate, offer_acceptance_rate)
    - profile_views_received_30d
    - connection_count

Design constraint from the spec: "Redrob should act only as a small
multiplier" and "should never dominate semantic similarity." This means
the multiplier must be tightly bounded around 1.0 — it nudges retrieval
confidence up or down, it does not re-rank candidates independently of
their actual relevance.

We bound the multiplier to the range [0.85, 1.05]:
    - A perfectly available, verified, fresh, highly-completed profile
      gets a small +5% boost.
    - A stale, unverified, unavailable profile gets at most a 15% penalty.
    - This range is intentionally narrow: even the worst-case redrob
      multiplier (0.85) cannot overcome a meaningful hybrid score gap,
      while the best case (1.05) cannot manufacture relevance that
      wasn't already there from dense + lexical retrieval.

This module performs NO retrieval, NO ranking, NO rejection. A candidate
is never dropped for redrob signal reasons at this stage — that filtering
already happened in the upstream Candidate Gate. Here, redrob only scales
an existing retrieval score.

CPU-only. No external APIs.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# 1. CONFIGURATION — bounded multiplier range and component weights
# ---------------------------------------------------------------------------

# Hard bounds on the final multiplier. Per spec, redrob must act only as a
# "small multiplier" that "should never dominate semantic similarity" — so
# this range is deliberately narrow around 1.0.
_MULTIPLIER_MIN: float = 0.85
_MULTIPLIER_MAX: float = 1.05

# Total deviation budget split across all five signal components, so that
# the multiplier reaches _MULTIPLIER_MAX ONLY when every single signal is
# simultaneously at its best, and _MULTIPLIER_MIN only when every signal is
# simultaneously at its worst. A candidate who is merely "good" on several
# signals (e.g. 87% completeness, 71% interview rate) must NOT be able to
# sum past a handful of individually-uncapped deltas and clip to the same
# ceiling as a literally perfect candidate — that would destroy the
# multiplier's ability to discriminate between "good" and "perfect"
# availability. Each component therefore gets an equal 1/5 share of the
# total bonus budget (+0.05 total) and total penalty budget (-0.15 total).
_TOTAL_BONUS_BUDGET: float = _MULTIPLIER_MAX - 1.0     # +0.05 distributed across 5 signals
_TOTAL_PENALTY_BUDGET: float = _MULTIPLIER_MIN - 1.0   # -0.15 distributed across 5 signals
_NUM_SIGNAL_COMPONENTS: int = 5

_PER_SIGNAL_MAX_BONUS: float = _TOTAL_BONUS_BUDGET / _NUM_SIGNAL_COMPONENTS      # +0.01
_PER_SIGNAL_MAX_PENALTY: float = _TOTAL_PENALTY_BUDGET / _NUM_SIGNAL_COMPONENTS  # -0.03

_OPEN_TO_WORK_BONUS: float = _PER_SIGNAL_MAX_BONUS
_OPEN_TO_WORK_PENALTY: float = _PER_SIGNAL_MAX_PENALTY

_COMPLETENESS_MAX_BONUS: float = _PER_SIGNAL_MAX_BONUS      # at completeness_score == 100
_COMPLETENESS_MAX_PENALTY: float = _PER_SIGNAL_MAX_PENALTY  # at completeness_score == 0

_VERIFICATION_BONUS_PER_CHANNEL: float = _PER_SIGNAL_MAX_BONUS / 2.0   # two channels share the budget
_VERIFICATION_PENALTY_NEITHER: float = _PER_SIGNAL_MAX_PENALTY

_INTERVIEW_COMPLETION_MAX_BONUS: float = _PER_SIGNAL_MAX_BONUS      # at rate == 1.0
_INTERVIEW_COMPLETION_MAX_PENALTY: float = _PER_SIGNAL_MAX_PENALTY  # at rate == 0.0

# Activity recency bands (days since last_active_date)
_RECENT_ACTIVITY_DAYS: int = 30
_MODERATE_ACTIVITY_DAYS: int = 90
_STALE_ACTIVITY_DAYS: int = 180

_RECENT_ACTIVITY_BONUS: float = _PER_SIGNAL_MAX_BONUS
_MODERATE_ACTIVITY_DELTA: float = 0.0
_STALE_ACTIVITY_PENALTY: float = _PER_SIGNAL_MAX_PENALTY * (2.0 / 3.0)
_VERY_STALE_ACTIVITY_PENALTY: float = _PER_SIGNAL_MAX_PENALTY
_UNKNOWN_ACTIVITY_PENALTY: float = _PER_SIGNAL_MAX_PENALTY / 3.0


# ---------------------------------------------------------------------------
# 2. DATE PARSING HELPER
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> date | None:
    """
    Parse an ISO-ish date string into a date object. Returns None on
    missing/unparseable input rather than raising, since a missing
    last_active_date should degrade gracefully (small penalty) rather
    than crash the retrieval pipeline for one candidate.
    """
    if not date_str:
        return None
    try:
        # Handles "YYYY-MM-DD" and full ISO timestamps alike.
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 3. PER-SIGNAL DELTA FUNCTIONS
# ---------------------------------------------------------------------------

def _open_to_work_delta(open_to_work_flag: Any) -> float:
    """Small bonus if actively open to work, small penalty otherwise."""
    if open_to_work_flag is True:
        return _OPEN_TO_WORK_BONUS
    if open_to_work_flag is False:
        return _OPEN_TO_WORK_PENALTY
    # Missing/unknown flag — neutral, no signal either way.
    return 0.0


def _completeness_delta(profile_completeness_score: Any) -> float:
    """
    Linearly interpolate between max penalty (score=0) and max bonus
    (score=100). Missing score is treated as the worst case (penalty),
    since an unscored profile cannot be assumed complete.
    """
    if profile_completeness_score is None:
        return _COMPLETENESS_MAX_PENALTY

    try:
        score = float(profile_completeness_score)
    except (TypeError, ValueError):
        return _COMPLETENESS_MAX_PENALTY

    score = max(0.0, min(100.0, score))
    fraction = score / 100.0

    return (
        _COMPLETENESS_MAX_PENALTY
        + fraction * (_COMPLETENESS_MAX_BONUS - _COMPLETENESS_MAX_PENALTY)
    )


def _verification_delta(verified_email: Any, verified_phone: Any) -> float:
    """Small bonus per verified channel; meaningful penalty if neither."""
    email_ok = verified_email is True
    phone_ok = verified_phone is True

    if not email_ok and not phone_ok:
        return _VERIFICATION_PENALTY_NEITHER

    delta = 0.0
    if email_ok:
        delta += _VERIFICATION_BONUS_PER_CHANNEL
    if phone_ok:
        delta += _VERIFICATION_BONUS_PER_CHANNEL
    return delta


def _interview_completion_delta(interview_completion_rate: Any) -> float:
    """
    Linearly interpolate between max penalty (rate=0) and max bonus
    (rate=1.0). Missing rate is treated as neutral (0.0 delta) rather
    than penalized, since many legitimate candidates may simply have no
    interview history yet (e.g. new to the platform) without that being
    a genuine availability concern.
    """
    if interview_completion_rate is None:
        return 0.0

    try:
        rate = float(interview_completion_rate)
    except (TypeError, ValueError):
        return 0.0

    rate = max(0.0, min(1.0, rate))

    return (
        _INTERVIEW_COMPLETION_MAX_PENALTY
        + rate * (_INTERVIEW_COMPLETION_MAX_BONUS - _INTERVIEW_COMPLETION_MAX_PENALTY)
    )


def _activity_recency_delta(
    last_active_date_str: Any,
    reference_date: date,
) -> float:
    """
    Bonus for recent activity, escalating penalty for staleness, small
    penalty for entirely missing/unparseable last_active_date.
    """
    parsed_date = _parse_date(last_active_date_str)
    if parsed_date is None:
        return _UNKNOWN_ACTIVITY_PENALTY

    days_inactive = (reference_date - parsed_date).days

    if days_inactive < 0:
        # Future-dated activity (clock skew / bad data) — treat as recent.
        return _RECENT_ACTIVITY_BONUS
    if days_inactive <= _RECENT_ACTIVITY_DAYS:
        return _RECENT_ACTIVITY_BONUS
    if days_inactive <= _MODERATE_ACTIVITY_DAYS:
        return _MODERATE_ACTIVITY_DELTA
    if days_inactive <= _STALE_ACTIVITY_DAYS:
        return _STALE_ACTIVITY_PENALTY
    return _VERY_STALE_ACTIVITY_PENALTY


# ---------------------------------------------------------------------------
# 4. TOP-LEVEL MULTIPLIER COMPUTATION
# ---------------------------------------------------------------------------

def compute_redrob_multiplier(
    redrob_signals: dict[str, Any],
    reference_date: date | None = None,
) -> float:
    """
    Compute the bounded Redrob confidence multiplier for one candidate
    from their allow-listed objective availability signals.

    Parameters
    ----------
    redrob_signals : dict
        The allow-listed redrob signals dict, as produced by
        candidate_document_builder.py's _extract_allowed_redrob_signals()
        (i.e. already restricted to: profile_completeness_score,
        open_to_work_flag, last_active_date, interview_completion_rate,
        verified_email, verified_phone). Any other keys present in this
        dict are silently ignored — this function never reads
        recruiter-popularity-style fields even if they were accidentally
        included upstream, as a defense-in-depth measure.
    reference_date : date | None
        "Today" for recency calculations. Defaults to the current UTC
        date. Exposed as a parameter for deterministic testing.

    Returns
    -------
    float
        Multiplier in [_MULTIPLIER_MIN, _MULTIPLIER_MAX] (default [0.85, 1.05]).
        Multiply this directly against the hybrid retrieval (RRF) score.
    """
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc).date()

    total_delta: float = 0.0

    total_delta += _open_to_work_delta(redrob_signals.get("open_to_work_flag"))
    total_delta += _completeness_delta(redrob_signals.get("profile_completeness_score"))
    total_delta += _verification_delta(
        redrob_signals.get("verified_email"),
        redrob_signals.get("verified_phone"),
    )
    total_delta += _interview_completion_delta(
        redrob_signals.get("interview_completion_rate")
    )
    total_delta += _activity_recency_delta(
        redrob_signals.get("last_active_date"), reference_date
    )

    multiplier = 1.0 + total_delta
    multiplier = max(_MULTIPLIER_MIN, min(_MULTIPLIER_MAX, multiplier))

    return multiplier


# ---------------------------------------------------------------------------
# 5. BATCH HELPER
# ---------------------------------------------------------------------------

def compute_redrob_multipliers_batch(
    candidate_redrob_signals: dict[str, dict[str, Any]],
    reference_date: date | None = None,
) -> dict[str, float]:
    """
    Compute Redrob multipliers for many candidates at once.

    Parameters
    ----------
    candidate_redrob_signals : dict[str, dict]
        Mapping of candidate_id -> allow-listed redrob_signals dict.
    reference_date : date | None
        "Today" for recency calculations, shared across the whole batch
        for consistency (so a multi-hour pipeline run doesn't apply
        different reference dates to different candidates).

    Returns
    -------
    dict[str, float]
        Mapping of candidate_id -> multiplier.
    """
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc).date()

    return {
        candidate_id: compute_redrob_multiplier(signals, reference_date=reference_date)
        for candidate_id, signals in candidate_redrob_signals.items()
    }