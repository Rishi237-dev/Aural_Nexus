"""
hybrid_retriever.py
=====================
Orchestrates the hybrid retrieval step:

    1. Dense retrieval  — FAISS IndexFlatIP search using the JD's E5
                           query embedding.
    2. Lexical retrieval — BM25 search using the JD's lexical query tokens.
    3. Fusion            — Reciprocal Rank Fusion (RRF) combines the two
                           independently-ranked lists. No manual weight
                           tuning, per the architecture spec.
    4. Redrob adjustment — the fused RRF score is multiplied by each
                           candidate's bounded Redrob confidence multiplier
                           (see redrob_adjustment.py) to produce the final
                           retrieval score.

This module does NOT build the FAISS/BM25 indexes (see build_faiss.py /
build_bm25.py) and does NOT generate embeddings (see embed_candidates.py).
It consumes already-built indexes and produces the final ranked retrieval
list of approximately 2,000-4,000 candidates.

This is retrieval, not ranking — no evidence scoring, no consistency
checks, no explanation generation, no career-progression scoring.

CPU-only. No external APIs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from retrieval.embedding.build_bm25 import CandidateBM25Index, tokenize_lexical_document
from retrieval.embedding.redrob_adjustment import compute_redrob_multiplier


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# How many results each individual retrieval method (dense, BM25) pulls
# before fusion. This should comfortably exceed the final target output
# size (2,000-4,000) because RRF fusion combines two independently-ranked
# lists — a candidate ranked highly by only ONE method still needs to
# appear in that method's top-K to be fusable at all. We retrieve a
# generous candidate pool per method (per spec: retrieval favors recall),
# then let RRF + redrob narrow down to the final output size.
DEFAULT_PER_METHOD_TOP_K: int = 6000

# Standard RRF damping constant. The canonical value from the original
# RRF paper (Cormack, Clarke & Buettcher, 2009) is k=60; this is a
# well-established default that avoids manual weight tuning, consistent
# with the architecture spec's explicit instruction not to hand-tune
# fusion weights.
RRF_K: int = 60

# Final output size range from the architecture spec.
DEFAULT_OUTPUT_TOP_K: int = 3000


# ---------------------------------------------------------------------------
# 2. OUTPUT DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    One candidate's full retrieval scoring breakdown.

    Attributes
    ----------
    candidate_id : str
    dense_score : float
        Raw cosine similarity from FAISS dense search. 0.0 if the
        candidate did not appear in the dense top-K at all.
    bm25_score : float
        Raw BM25 score from lexical search. 0.0 if the candidate did not
        appear in the BM25 top-K at all.
    rrf_score : float
        Reciprocal Rank Fusion score combining both ranked lists.
    redrob_multiplier : float
        Bounded [0.85, 1.05] confidence multiplier from objective
        availability signals.
    final_retrieval_score : float
        rrf_score × redrob_multiplier. This is the sort key for the
        final output list.
    """
    candidate_id: str
    dense_score: float
    bm25_score: float
    rrf_score: float
    redrob_multiplier: float
    final_retrieval_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-safe dict."""
        return {
            "candidate_id": self.candidate_id,
            "dense_score": round(self.dense_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "redrob_multiplier": round(self.redrob_multiplier, 6),
            "final_retrieval_score": round(self.final_retrieval_score, 6),
        }


# ---------------------------------------------------------------------------
# 3. RECIPROCAL RANK FUSION
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Fuse two independently-ranked result lists using Reciprocal Rank Fusion.

    RRF formula for a candidate appearing at rank r (1-indexed) in a
    ranked list:
        rrf_contribution = 1 / (k + r)

    A candidate's total RRF score is the sum of its contributions across
    all lists it appears in. A candidate absent from a list simply
    contributes 0 from that list (it is NOT penalized beyond simply not
    receiving that list's contribution).

    This is rank-based, not score-based — RRF deliberately ignores the
    raw score magnitudes from each method (which are on incomparable
    scales: cosine similarity in [-1, 1] vs. unbounded BM25 scores) and
    fuses based purely on each candidate's position within each ranked
    list. This is exactly why RRF requires no manual weight tuning
    between dense and lexical scores, per the architecture spec.

    Parameters
    ----------
    dense_results : list[tuple[str, float]]
        (candidate_id, score) pairs from FAISS dense search, already
        sorted descending by score (as returned by build_faiss.dense_search()).
    bm25_results : list[tuple[str, float]]
        (candidate_id, score) pairs from BM25 search, already sorted
        descending by score (as returned by CandidateBM25Index.search()).
    k : int
        RRF damping constant. Higher k reduces the influence of exact
        rank position (flattens the curve); lower k makes top ranks
        dominate more sharply. Default 60 is the standard literature value.

    Returns
    -------
    dict[str, float]
        Mapping of candidate_id -> total RRF score, for every candidate
        that appeared in at least one of the two input lists.
    """
    rrf_scores: dict[str, float] = {}

    for rank, (candidate_id, _score) in enumerate(dense_results, start=1):
        rrf_scores[candidate_id] = rrf_scores.get(candidate_id, 0.0) + 1.0 / (k + rank)

    for rank, (candidate_id, _score) in enumerate(bm25_results, start=1):
        rrf_scores[candidate_id] = rrf_scores.get(candidate_id, 0.0) + 1.0 / (k + rank)

    return rrf_scores


# ---------------------------------------------------------------------------
# 4. QUERY EMBEDDING — E5 query-side encoding for the JD
# ---------------------------------------------------------------------------

# E5's model card recommends an instruction prefix for the QUERY side
# in asymmetric retrieval (query vs. passage) to improve retrieval quality.
# Candidate documents (passages) get NO prefix (see embed_candidates.py);
# only the JD (query) gets this prefix. This asymmetry is intentional and
# matches E5's documented usage pattern, not an architectural addition —
# it is purely an embedding-input-formatting detail of the E5 model itself.
E5_QUERY_INSTRUCTION_PREFIX: str = (
    "Represent this sentence for searching relevant passages: "
)


def embed_jd_query(jd_semantic_document: str, model: Any) -> np.ndarray:
    """
    Encode the JD's semantic document into a normalized E5 query embedding.

    Parameters
    ----------
    jd_semantic_document : str
        The JD's semantic_document (from jd_document_builder.py).
    model : SentenceTransformer
        A loaded E5 model instance (e.g. from
        embed_candidates._get_e5_model()).

    Returns
    -------
    np.ndarray, shape (768,), dtype float32
        L2-normalized query embedding.
    """
    prefixed_query = E5_QUERY_INSTRUCTION_PREFIX + jd_semantic_document

    # Use the device the model is already loaded on (cuda:0 or cpu).
    # SentenceTransformer stores this as model.device; we read it here
    # rather than re-importing torch so this module stays CPU-focused.
    model_device = str(getattr(model, "device", "cpu"))

    embedding = model.encode(
        [prefixed_query],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=model_device,
    )
    return embedding[0].astype(np.float32)


# ---------------------------------------------------------------------------
# 5. TOP-LEVEL HYBRID RETRIEVAL ORCHESTRATION
# ---------------------------------------------------------------------------

def run_hybrid_retrieval(
    faiss_index: "faiss.Index",  # noqa: F821
    faiss_candidate_ids: list[str],
    bm25_index: CandidateBM25Index,
    jd_query_embedding: np.ndarray,
    jd_lexical_document: str,
    candidate_redrob_signals: dict[str, dict[str, Any]],
    per_method_top_k: int = DEFAULT_PER_METHOD_TOP_K,
    output_top_k: int = DEFAULT_OUTPUT_TOP_K,
    rrf_k: int = RRF_K,
) -> list[RetrievalResult]:
    """
    Run the complete hybrid retrieval pipeline: dense search + BM25 search
    + RRF fusion + Redrob adjustment, returning the final ranked output list.

    Parameters
    ----------
    faiss_index : faiss.Index
        Loaded IndexFlatIP index (from build_faiss.load_faiss_index()).
    faiss_candidate_ids : list[str]
        candidate_ids in FAISS row order (from
        build_faiss.build_and_persist_faiss_index() or
        embed_candidates.load_candidate_embeddings()).
    bm25_index : CandidateBM25Index
        Loaded BM25 index (from build_bm25.load_bm25_index()).
    jd_query_embedding : np.ndarray, shape (768,)
        L2-normalized JD query embedding (from embed_jd_query()).
    jd_lexical_document : str
        The JD's lexical_document string (from jd_document_builder.py),
        will be tokenized internally.
    candidate_redrob_signals : dict[str, dict]
        Mapping of candidate_id -> allow-listed redrob_signals dict (from
        candidate_document_builder.py's output, as persisted in
        candidate_documents.jsonl by embed_candidates.py).
    per_method_top_k : int
        How many results each individual method (dense, BM25) retrieves
        before fusion. Should exceed output_top_k to give RRF a wide
        enough pool, since a candidate must appear in at least one
        method's top-K to be retrievable at all.
    output_top_k : int
        Final number of candidates to return after fusion + redrob
        adjustment, sorted by final_retrieval_score descending.
    rrf_k : int
        RRF damping constant (default 60, standard literature value).

    Returns
    -------
    list[RetrievalResult]
        Top output_top_k candidates, sorted descending by
        final_retrieval_score. Length is
        min(output_top_k, number of unique candidates retrieved by
        dense ∪ BM25).
    """
    from retrieval.embedding.build_faiss import dense_search

    # ---- Step 1: Dense retrieval ----
    print(
        f"[hybrid_retriever] Running dense search (top_k={per_method_top_k:,})...",
        file=sys.stderr,
    )
    dense_results = dense_search(
        faiss_index, jd_query_embedding, faiss_candidate_ids, top_k=per_method_top_k
    )
    dense_score_lookup: dict[str, float] = dict(dense_results)

    # ---- Step 2: Lexical (BM25) retrieval ----
    print(
        f"[hybrid_retriever] Running BM25 search (top_k={per_method_top_k:,})...",
        file=sys.stderr,
    )
    jd_query_tokens = tokenize_lexical_document(jd_lexical_document)
    bm25_results = bm25_index.search(jd_query_tokens, top_k=per_method_top_k)
    bm25_score_lookup: dict[str, float] = dict(bm25_results)

    print(
        f"[hybrid_retriever] Dense: {len(dense_results):,} results, "
        f"BM25: {len(bm25_results):,} results.",
        file=sys.stderr,
    )

    # ---- Step 3: Reciprocal Rank Fusion ----
    rrf_scores = reciprocal_rank_fusion(dense_results, bm25_results, k=rrf_k)

    print(
        f"[hybrid_retriever] RRF fusion produced {len(rrf_scores):,} unique "
        f"candidates (union of dense and BM25 result sets).",
        file=sys.stderr,
    )

    # ---- Step 4: Redrob adjustment + final scoring ----
    results: list[RetrievalResult] = []

    for candidate_id, rrf_score in rrf_scores.items():
        redrob_signals = candidate_redrob_signals.get(candidate_id, {})
        multiplier = compute_redrob_multiplier(redrob_signals)

        final_score = rrf_score * multiplier

        results.append(RetrievalResult(
            candidate_id=candidate_id,
            dense_score=dense_score_lookup.get(candidate_id, 0.0),
            bm25_score=bm25_score_lookup.get(candidate_id, 0.0),
            rrf_score=rrf_score,
            redrob_multiplier=multiplier,
            final_retrieval_score=final_score,
        ))

    # ---- Step 5: Sort and truncate to output_top_k ----
    results.sort(key=lambda r: r.final_retrieval_score, reverse=True)
    final_results = results[:output_top_k]

    print(
        f"[hybrid_retriever] Returning top {len(final_results):,} candidates "
        f"(requested output_top_k={output_top_k:,}).",
        file=sys.stderr,
    )

    return final_results