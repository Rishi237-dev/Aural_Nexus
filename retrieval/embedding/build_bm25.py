"""
build_bm25.py
==============
Builds, persists, and queries the BM25 lexical retrieval index over
candidate lexical_documents.

Per architecture spec:
    - Library: rank_bm25
    - Input: the lexical_document produced by candidate_document_builder.py
      (dense, non-natural-language token soup optimized for exact matching).

BM25Okapi (rank_bm25's standard implementation) is used. The index is
built from the same tokenization scheme used to construct the lexical
documents in the first place — both candidate_document_builder.py and
jd_document_builder.py already produce space-separated token strings, so
tokenization here is simply str.split().

Persistence
------------
rank_bm25's BM25Okapi object is a lightweight pure-Python/NumPy structure
(term-frequency tables + corpus statistics). It is persisted via pickle,
which is safe here because:
    (a) this is an internal pipeline artifact, never loaded from
        untrusted/external sources, and
    (b) rank_bm25 has no native serialization format of its own.

The pickle file is written/read only by this module; candidate_ids
ordering is tracked and persisted alongside it (mirroring build_faiss.py's
approach) since BM25Okapi has no concept of document IDs — it only knows
document positions.

This module does NOT generate embeddings (see embed_candidates.py) and
does NOT perform dense search or fusion (see build_faiss.py /
hybrid_retriever.py). It only builds/persists/queries the BM25 index.

CPU-only. No external APIs.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

from retrieval.embedding.artifact_io import load_candidate_documents


# ---------------------------------------------------------------------------
# 1. TOKENIZATION
# ---------------------------------------------------------------------------

def tokenize_lexical_document(lexical_document: str) -> list[str]:
    """
    Tokenize a lexical_document string into a list of tokens for BM25.

    Both candidate_document_builder.py and jd_document_builder.py already
    produce clean, space-separated, lowercased token strings (technology
    names, skill names, decomposed words). The only tokenization needed
    here is a plain whitespace split — re-applying regex tokenization
    would double-process text that's already been through
    _tokenize_for_lexical() upstream and could silently diverge from the
    document-builder tokenization scheme.

    Parameters
    ----------
    lexical_document : str
        Space-separated token string from a lexical document builder.

    Returns
    -------
    list[str]
        Token list, ready for BM25Okapi.
    """
    if not lexical_document:
        return []
    return lexical_document.split()


# ---------------------------------------------------------------------------
# 2. BM25 INDEX CONTAINER
# ---------------------------------------------------------------------------

class CandidateBM25Index:
    """
    Wraps a rank_bm25.BM25Okapi index together with the candidate_ids
    ordering, since BM25Okapi itself only knows positional document
    indices and has no notion of an external ID.

    Attributes
    ----------
    bm25 : BM25Okapi
        The underlying rank_bm25 index.
    candidate_ids : list[str]
        candidate_id at position i corresponds to the document that was
        at position i when the index was built (i.e. corpus[i]).
    """

    def __init__(self, bm25: Any, candidate_ids: list[str]) -> None:
        self.bm25 = bm25
        self.candidate_ids = candidate_ids

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        """
        Score the corpus against query_tokens and return the top_k results.

        Parameters
        ----------
        query_tokens : list[str]
            Tokenized query (typically the JD's lexical_document, tokenized
            via tokenize_lexical_document()).
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[tuple[str, float]]
            (candidate_id, bm25_score) pairs, sorted descending by score.
            BM25 scores are unbounded non-negative floats (not normalized
            to [0, 1] — that normalization, if needed, happens in
            hybrid_retriever.py / RRF fusion, which is rank-based and does
            not require comparable score scales across dense/lexical sides).
        """
        if not query_tokens or len(self.candidate_ids) == 0:
            return []

        scores = self.bm25.get_scores(query_tokens)  # np.ndarray, shape (N,)

        effective_k = min(top_k, len(self.candidate_ids))

        # argpartition for efficiency on large corpora (85k candidates),
        # then sort only the top_k slice descending.
        if effective_k >= len(scores):
            top_indices = scores.argsort()[::-1]
        else:
            partitioned = scores.argpartition(-effective_k)[-effective_k:]
            top_indices = partitioned[scores[partitioned].argsort()[::-1]]

        results: list[tuple[str, float]] = [
            (self.candidate_ids[idx], float(scores[idx]))
            for idx in top_indices[:effective_k]
        ]
        return results


# ---------------------------------------------------------------------------
# 3. INDEX CONSTRUCTION
# ---------------------------------------------------------------------------

def build_bm25_index(
    candidate_ids: list[str],
    lexical_documents: list[str],
) -> CandidateBM25Index:
    """
    Build a BM25Okapi index from parallel lists of candidate_ids and
    lexical_documents.

    Parameters
    ----------
    candidate_ids : list[str]
        candidate_id at position i.
    lexical_documents : list[str]
        lexical_document (space-separated token string) at position i,
        corresponding to candidate_ids[i].

    Returns
    -------
    CandidateBM25Index
    """
    if len(candidate_ids) != len(lexical_documents):
        raise ValueError(
            f"candidate_ids and lexical_documents must have equal length "
            f"(got {len(candidate_ids)} vs {len(lexical_documents)})."
        )

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank_bm25 is required to build the BM25 index. Install it or run the pipeline in lexical-only fallback mode."
        ) from exc

    tokenized_corpus: list[list[str]] = [
        tokenize_lexical_document(doc) for doc in lexical_documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return CandidateBM25Index(bm25=bm25, candidate_ids=list(candidate_ids))


# ---------------------------------------------------------------------------
# 4. PERSISTENCE
# ---------------------------------------------------------------------------

def save_bm25_index(index: CandidateBM25Index, index_path: str | Path) -> None:
    """
    Persist a CandidateBM25Index to disk via pickle.

    Parameters
    ----------
    index : CandidateBM25Index
        The built index.
    index_path : str | Path
        Destination file path (e.g. candidate_bm25.pkl).
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_bm25_index(index_path: str | Path) -> CandidateBM25Index:
    """
    Load a previously persisted CandidateBM25Index from disk.

    Parameters
    ----------
    index_path : str | Path
        Path to the persisted pickle file.

    Returns
    -------
    CandidateBM25Index
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {index_path}. "
            "Run build_and_persist_bm25_index() first."
        )
    with index_path.open("rb") as fh:
        index = pickle.load(fh)
    return index


# ---------------------------------------------------------------------------
# 5. TOP-LEVEL ORCHESTRATION — build from persisted documents and save
# ---------------------------------------------------------------------------

def build_and_persist_bm25_index(
    embeddings_dir: str | Path,
    index_path: str | Path,
) -> CandidateBM25Index:
    """
    Stream persisted candidate documents (from embed_candidates.py's
    candidate_documents.jsonl output), build the BM25 index over their
    lexical_documents, and persist it to disk.

    Note: unlike build_faiss.py, this does NOT support incremental
    no-recompute updates — BM25's IDF statistics are corpus-wide, so
    adding new documents changes the relevance of every existing document.
    The index is always rebuilt fully from the current candidate_documents.jsonl.
    This is computationally cheap (BM25Okapi construction over 85k short
    token lists is a matter of seconds), unlike the E5 embedding step
    which is comparatively expensive — hence why only embeddings get the
    no-recompute treatment.

    Parameters
    ----------
    embeddings_dir : str | Path
        Directory containing candidate_documents.jsonl (from embed_candidates.py).
    index_path : str | Path
        Destination path for the persisted BM25 index (e.g. candidate_bm25.pkl).

    Returns
    -------
    CandidateBM25Index
    """
    candidate_ids: list[str] = []
    lexical_documents: list[str] = []

    for record in load_candidate_documents(embeddings_dir):
        candidate_ids.append(record["candidate_id"])
        lexical_documents.append(record.get("lexical_document", ""))

    if not candidate_ids:
        raise ValueError(
            f"No candidate documents found in {embeddings_dir}. "
            "Run generate_candidate_embeddings() first."
        )

    print(
        f"[build_bm25] Building BM25 index over {len(candidate_ids):,} "
        f"candidate lexical documents...",
        file=sys.stderr,
    )

    index = build_bm25_index(candidate_ids, lexical_documents)
    save_bm25_index(index, index_path)

    print(
        f"[build_bm25] Persisted BM25 index ({len(candidate_ids):,} documents) "
        f"to {index_path}",
        file=sys.stderr,
    )

    return index


# ---------------------------------------------------------------------------
# 6. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Command-line interface for BM25 index construction.

    Usage:
        python -m retrieval.build_bm25 \\
            ./retrieval_artifacts \\
            ./retrieval_artifacts/candidate_bm25.pkl
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and persist the BM25 lexical retrieval index."
    )
    parser.add_argument(
        "embeddings_dir",
        help="Directory containing candidate_documents.jsonl (from embed_candidates.py).",
    )
    parser.add_argument(
        "index_path",
        help="Destination path for the persisted BM25 index (e.g. candidate_bm25.pkl).",
    )

    args = parser.parse_args()
    build_and_persist_bm25_index(args.embeddings_dir, args.index_path)


if __name__ == "__main__":
    _cli()