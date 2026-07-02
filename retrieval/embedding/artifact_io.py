"""Lightweight helpers for persisted retrieval artifacts.

This module intentionally avoids importing torch, sentence-transformers,
faiss, or rank_bm25 so that downstream modules can load persisted
artifacts without paying the heavy ML import cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def candidate_embeddings_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "candidate_embeddings.npz"


def candidate_documents_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "candidate_documents.jsonl"


def load_candidate_embeddings(output_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Load persisted candidate embeddings for downstream retrieval."""
    npz_path = candidate_embeddings_path(output_dir)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No embedding archive found at {npz_path}. Run generate_candidate_embeddings() first."
        )

    with np.load(npz_path, allow_pickle=False) as data:
        candidate_ids = [str(cid) for cid in data["candidate_ids"].tolist()]
        embeddings = data["embeddings"].astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embedding matrix shape in {npz_path}: {embeddings.shape!r}")

    return candidate_ids, embeddings


def load_candidate_documents(output_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Stream persisted candidate document metadata from JSONL."""
    documents_path = candidate_documents_path(output_dir)
    if not documents_path.exists():
        raise FileNotFoundError(
            f"No document metadata found at {documents_path}. Run generate_candidate_embeddings() first."
        )

    with documents_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)