"""
build_faiss.py
================
Builds, persists, and queries the FAISS dense retrieval index over
candidate E5 embeddings.

Per architecture spec:
    - Index type: IndexFlatIP (exact inner-product search)
    - Embeddings must be L2-normalized before indexing, so that inner
      product == cosine similarity (normalization is already performed
      in embed_candidates.py at encode time; this module asserts that
      invariant rather than re-normalizing, to fail loudly if it's ever
      violated upstream).
    - Persisted as candidate.index on disk.

IndexFlatIP performs exact (non-approximate) search. For ~85k candidates
at 768 dimensions this is entirely tractable on CPU (a few hundred MB of
memory, sub-second query latency) — no need for an approximate index
(IVF/HNSW) at this scale, and the spec explicitly recommends IndexFlatIP.

This module does NOT generate embeddings (see embed_candidates.py) and
does NOT perform BM25 or fusion (see build_bm25.py / hybrid_retriever.py).
It only builds/persists/queries the dense vector index.

CPU-only. No external APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from retrieval.embedding.artifact_io import load_candidate_embeddings


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Tolerance for the L2-normalization invariant check. E5 embeddings with
# normalize_embeddings=True should have norm extremely close to 1.0; we
# allow a small float32 tolerance rather than requiring exact equality.
_NORMALIZATION_TOLERANCE: float = 1e-3


# ---------------------------------------------------------------------------
# 2. INDEX CONSTRUCTION
# ---------------------------------------------------------------------------

def _assert_normalized(embeddings: np.ndarray) -> None:
    """
    Verify that embeddings are (approximately) L2-normalized.

    This is a hard precondition for IndexFlatIP to behave as cosine
    similarity. If embed_candidates.py is ever changed to skip
    normalization, this assertion fails loudly here rather than silently
    producing wrong (raw dot-product, not cosine) similarity scores
    three modules downstream.
    """
    if embeddings.shape[0] == 0:
        return

    norms = np.linalg.norm(embeddings, axis=1)
    max_deviation = float(np.max(np.abs(norms - 1.0)))

    if max_deviation > _NORMALIZATION_TOLERANCE:
        raise ValueError(
            f"Embeddings are not L2-normalized (max deviation from norm=1.0 "
            f"is {max_deviation:.6f}, tolerance is {_NORMALIZATION_TOLERANCE}). "
            "IndexFlatIP requires normalized vectors for inner product to "
            "equal cosine similarity. Check embed_candidates.py's "
            "normalize_embeddings=True setting."
        )


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":  # noqa: F821
    """
    Build an IndexFlatIP FAISS index from a matrix of L2-normalized
    candidate embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D), dtype float32
        L2-normalized E5 embeddings (one row per candidate, in the same
        order as the corresponding candidate_ids list).

    Returns
    -------
    faiss.IndexFlatIP
        Index ready for search(). Inner product on normalized vectors
        equals cosine similarity.
    """
    import faiss  # imported lazily so other modules don't pay the import cost

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    _assert_normalized(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # FAISS requires C-contiguous float32 arrays.
    embeddings_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(embeddings_contiguous)

    return index


# ---------------------------------------------------------------------------
# 3. PERSISTENCE
# ---------------------------------------------------------------------------

def save_faiss_index(index: "faiss.Index", index_path: str | Path) -> None:  # noqa: F821
    """
    Persist a FAISS index to disk as candidate.index (or the given path).

    Parameters
    ----------
    index : faiss.Index
        The constructed index (from build_faiss_index()).
    index_path : str | Path
        Destination file path.
    """
    import faiss

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: str | Path) -> "faiss.Index":  # noqa: F821
    """
    Load a previously persisted FAISS index from disk.

    Parameters
    ----------
    index_path : str | Path
        Path to the persisted candidate.index file.

    Returns
    -------
    faiss.Index
    """
    import faiss

    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_path}. "
            "Run build_and_persist_faiss_index() first."
        )
    return faiss.read_index(str(index_path))


# ---------------------------------------------------------------------------
# 4. TOP-LEVEL ORCHESTRATION — build from persisted embeddings and save
# ---------------------------------------------------------------------------

def build_and_persist_faiss_index(
    embeddings_dir: str | Path,
    index_path: str | Path,
) -> list[str]:
    """
    Load persisted candidate embeddings (from embed_candidates.py's output),
    build the FAISS IndexFlatIP index, and persist it to disk.

    The FAISS index itself stores vectors only — NOT candidate_ids. FAISS
    returns integer row positions on search, so the caller must separately
    persist/track the candidate_ids list in the same order the embeddings
    were added, to translate row positions back to candidate_ids. This
    function returns that ordered list so the orchestrating pipeline can
    persist it alongside the index.

    Parameters
    ----------
    embeddings_dir : str | Path
        Directory containing candidate_embeddings.npz (from embed_candidates.py).
    index_path : str | Path
        Destination path for the persisted FAISS index (e.g. candidate.index).

    Returns
    -------
    list[str]
        candidate_ids in the exact order corresponding to FAISS row indices
        0..N-1. The caller is responsible for persisting this ordering
        (e.g. as a sidecar JSON/text file) if needed across process restarts;
        hybrid_retriever.py re-derives it directly from
        load_candidate_embeddings() to guarantee it always matches the index
        build order, rather than trusting a separately persisted copy.
    """
    candidate_ids, embeddings = load_candidate_embeddings(embeddings_dir)

    if len(candidate_ids) == 0:
        raise ValueError(
            f"No embeddings found in {embeddings_dir}. "
            "Run generate_candidate_embeddings() first."
        )

    print(
        f"[build_faiss] Building IndexFlatIP over {len(candidate_ids):,} "
        f"candidate embeddings (dim={embeddings.shape[1]})...",
        file=sys.stderr,
    )

    index = build_faiss_index(embeddings)
    save_faiss_index(index, index_path)

    print(
        f"[build_faiss] Persisted FAISS index ({index.ntotal:,} vectors) to "
        f"{index_path}",
        file=sys.stderr,
    )

    return candidate_ids


# ---------------------------------------------------------------------------
# 5. DENSE SEARCH
# ---------------------------------------------------------------------------

def dense_search(
    index: "faiss.Index",  # noqa: F821
    query_embedding: np.ndarray,
    candidate_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Perform a dense (cosine-similarity-via-inner-product) search against
    the FAISS index for a single query embedding.

    Parameters
    ----------
    index : faiss.Index
        The built/loaded IndexFlatIP index.
    query_embedding : np.ndarray, shape (D,) or (1, D), dtype float32
        L2-normalized query embedding (the JD's semantic embedding).
        Must use the SAME normalization convention as the indexed
        candidate embeddings (normalize_embeddings=True at encode time).
    candidate_ids : list[str]
        Ordered candidate_ids matching the index's row order, as returned
        by build_and_persist_faiss_index() / load_candidate_embeddings().
    top_k : int
        Number of nearest neighbors to retrieve.

    Returns
    -------
    list[tuple[str, float]]
        (candidate_id, cosine_similarity_score) pairs, sorted descending
        by score. Length is min(top_k, index.ntotal).
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
    _assert_normalized(query_embedding)

    effective_k = min(top_k, index.ntotal)
    if effective_k == 0:
        return []

    scores, row_indices = index.search(query_embedding, effective_k)

    # scores/row_indices have shape (1, effective_k) for a single query
    scores = scores[0]
    row_indices = row_indices[0]

    results: list[tuple[str, float]] = []
    for row_idx, score in zip(row_indices, scores):
        if row_idx < 0:
            # FAISS returns -1 for unfilled slots when fewer than top_k
            # results exist; skip these.
            continue
        results.append((candidate_ids[row_idx], float(score)))

    return results


# ---------------------------------------------------------------------------
# 6. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Command-line interface for FAISS index construction.

    Usage:
        python -m retrieval.build_faiss \\
            ./retrieval_artifacts \\
            ./retrieval_artifacts/candidate.index
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and persist the FAISS IndexFlatIP dense retrieval index."
    )
    parser.add_argument(
        "embeddings_dir",
        help="Directory containing candidate_embeddings.npz (from embed_candidates.py).",
    )
    parser.add_argument(
        "index_path",
        help="Destination path for the persisted FAISS index (e.g. candidate.index).",
    )

    args = parser.parse_args()
    build_and_persist_faiss_index(args.embeddings_dir, args.index_path)


if __name__ == "__main__":
    _cli()