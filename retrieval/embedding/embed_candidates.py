"""
embed_candidates.py
=====================
Generates offline E5 embeddings for every candidate's semantic document
and persists them to disk in a streaming-safe, idempotent way.

Model: intfloat/e5-base-v2 (via sentence-transformers).

Persistence format
-------------------
Embeddings are persisted as a single compressed NumPy archive plus a
parallel JSONL metadata file:

    candidate_embeddings.npz
        - candidate_ids : np.ndarray[str], shape (N,)
        - embeddings    : np.ndarray[float32], shape (N, D)
        - model_name    : np.ndarray[str], shape ()

    candidate_documents.jsonl
        One line per candidate:
        {"candidate_id": ..., "semantic_document": ..., "lexical_document": ..., 
         "redrob_signals": {...}}

This split exists because NumPy archives are efficient for the embedding
matrix (used directly by FAISS) while JSONL is efficient for streaming
text metadata without loading it all into memory at once.

No-recompute guarantee
------------------------
Before encoding, we check candidate_embeddings.npz for existing
candidate_ids. Only candidates NOT already present are encoded and
appended. This makes re-running the pipeline on an updated
filtered_candidates.jsonl cheap — already-embedded candidates are
skipped entirely.

Device selection
-----------------
Embedding generation uses CUDA (cuda:0) when available, falling back
to CPU automatically. All other retrieval operations (BM25, FAISS
indexing, document building, file I/O, orchestration) remain CPU-only
and are unaffected by this selection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# CPU / backend tuning — MUST be set before torch / transformers import.
# Without USE_TF=0, transformers pulls in TensorFlow (~2s+ import, extra RAM,
# and lower CPU throughput). This was the main cause of slow encoding.
# ---------------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 8
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))

import numpy as np
from tqdm import tqdm

_torch_initialized = False

def _init_torch() -> None:
    """
    One-time torch initialisation.

    CPU thread counts are only tuned when CUDA is unavailable — on a GPU
    run, inter-op parallelism is handled by CUDA kernels, and setting
    CPU thread counts can interfere with CUDA stream scheduling.
    """
    global _torch_initialized
    if _torch_initialized:
        return
    import torch
    if not torch.cuda.is_available():
        # Tune CPU threads only when the CPU is the inference backend.
        torch.set_num_threads(_CPU_COUNT)
        torch.set_num_interop_threads(min(4, max(1, _CPU_COUNT // 4)))
        print(
            f"[embed_candidates] CPU mode — torch threads: "
            f"intra={torch.get_num_threads()}",
            file=sys.stderr,
        )
    _torch_initialized = True


from retrieval.embedding.candidate_document_builder import (
    CandidateRetrievalDocument,
    build_candidate_retrieval_document,
)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Model identifier — pinned per architecture spec. Do not change without
# re-embedding the entire candidate pool (embedding spaces are not
# cross-compatible between model versions).
E5_MODEL_NAME: str = "intfloat/e5-base-v2"

# Batch size for encoding — larger batches improve GPU utilisation and throughput.
# Increased to 192 for better GPU efficiency with FP16 mixed precision.
DEFAULT_BATCH_SIZE: int = 192

# Rough chars-per-token heuristic for pre-truncating before tokenization.
# The model's max_seq_length is typically 512; capping input strings avoids
# tokenizing very long career descriptions that would be truncated anyway.
_CHARS_PER_TOKEN: int = 4

# E5 instruction prefixes for asymmetric retrieval.
# Candidate documents are passage inputs; JD queries are query inputs.
PASSAGE_INSTRUCTION_PREFIX: str = "passage: "


# ---------------------------------------------------------------------------
# 2. LAZY MODEL LOADING
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}


def _get_embedding_device() -> str:
    """
    Resolve the embedding device.

    Returns ``"cuda:0"`` when CUDA is available (RTX 3050 or any CUDA
    device at index 0), otherwise falls back to ``"cpu"``.
    The explicit ``cuda:0`` spec is accepted directly by SentenceTransformer
    and avoids device-index ambiguity in multi-GPU environments.
    """
    _init_torch()
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _log_device_info(device: str, batch_size: int) -> None:
    """
    Print a startup banner showing the selected device, GPU name (if any),
    CUDA availability, and the batch size that will be used for encoding.

    This runs once per pipeline invocation, immediately before the model
    is loaded, so the information is visible before the slow model-load
    step begins.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    if cuda_available and device.startswith("cuda"):
        device_index = int(device.split(":")[1]) if ":" in device else 0
        gpu_name = torch.cuda.get_device_name(device_index)
    else:
        gpu_name = "N/A"

    print(
        f"[embed_candidates] ─── Device Selection ───────────────────────────",
        file=sys.stderr,
    )
    print(f"[embed_candidates]   Selected device : {device}", file=sys.stderr)
    print(f"[embed_candidates]   GPU name        : {gpu_name}", file=sys.stderr)
    print(f"[embed_candidates]   CUDA available  : {cuda_available}", file=sys.stderr)
    print(f"[embed_candidates]   Batch size      : {batch_size}", file=sys.stderr)
    print(
        f"[embed_candidates] ─────────────────────────────────────────────────",
        file=sys.stderr,
    )


def _get_e5_model(model_name: str = E5_MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE) -> Any:
    """
    Lazily load and cache the SentenceTransformer model.

    Logs device selection, GPU name, CUDA availability, and batch size
    before loading, so the information is visible before the slow
    model-load step begins.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    device = _get_embedding_device()
    _log_device_info(device, batch_size)
    from sentence_transformers import SentenceTransformer

    print(f"[embed_candidates] Loading model '{model_name}' on {device}...", file=sys.stderr)
    try:
        model = SentenceTransformer(model_name, device=device)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to load SentenceTransformer model '{model_name}'. The local Hugging Face cache may be incomplete or corrupted. Remove the partially downloaded model folder and retry, or choose another model via --model."
        ) from exc
    model.eval()
    # Enable FP16 (mixed precision) for ~2x GPU speedup with negligible quality loss.
    # E5 inference in FP16 is safe and widely used in production.
    model.half()
    # Ensure model parameters are explicitly moved to the selected device
    try:
        import torch
        model.to(torch.device(device))
    except Exception:
        # Some SentenceTransformer backends may already have placed the
        # model on the requested device; ignore failures to be robust.
        pass

    # Warmup a single encode on the chosen device to trigger any lazy
    # initialisation (tokenizers, CUDA kernels, etc.). Passing `device`
    # here is redundant with model.to(...), but kept for clarity.
    model.encode(
        ["warmup"],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )
    _model_cache[model_name] = model
    print(
        f"[embed_candidates] Model ready (max_seq_length={model.get_max_seq_length()}).",
        file=sys.stderr,
    )
    return model


# ---------------------------------------------------------------------------
# 3. PERSISTED STATE — load existing embeddings to support no-recompute
# ---------------------------------------------------------------------------


def _load_existing_embedding_ids(npz_path: Path, required_model_name: str | None = None) -> set[str]:
    """Return candidate_ids already persisted in the embedding archive."""
    if not npz_path.exists():
        return set()

    with np.load(npz_path, allow_pickle=False) as data:
        existing_ids = data["candidate_ids"]
        persisted_model_name = str(data["model_name"].tolist())

    if required_model_name is not None and persisted_model_name != required_model_name:
        raise ValueError(
            f"Embedding archive model mismatch: found '{persisted_model_name}', "
            f"expected '{required_model_name}'. Rebuild the archive or use the "
            "matching model."
        )

    return {str(cid) for cid in existing_ids.tolist()}



def _load_existing_embeddings(npz_path: Path, required_model_name: str | None = None) -> tuple[list[str], np.ndarray]:
    """Load the existing embedding archive as (candidate_ids, embeddings)."""
    if not npz_path.exists():
        return [], np.zeros((0, 0), dtype=np.float32)

    with np.load(npz_path, allow_pickle=False) as data:
        persisted_model_name = str(data["model_name"].tolist())
        candidate_ids = [str(cid) for cid in data["candidate_ids"].tolist()]
        embeddings = data["embeddings"].astype(np.float32)

    if required_model_name is not None and persisted_model_name != required_model_name:
        raise ValueError(
            f"Embedding archive model mismatch: found '{persisted_model_name}', "
            f"expected '{required_model_name}'. Rebuild the archive or use the "
            "matching model."
        )

    return candidate_ids, embeddings


# ---------------------------------------------------------------------------
# 4. STREAMING CANDIDATE READER
# ---------------------------------------------------------------------------


def _stream_filtered_candidates(jsonl_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one candidate record at a time from filtered_candidates.jsonl."""
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[embed_candidates] WARNING: Skipping malformed JSON "
                    f"on line {line_num}: {exc}",
                    file=sys.stderr,
                )


def _count_filtered_candidate_lines(jsonl_path: str | Path) -> int:
    """Count non-empty lines for tqdm's total."""
    path = Path(jsonl_path)
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# 5. BATCH ENCODING
# ---------------------------------------------------------------------------


def _truncate_texts_for_model(model: Any, texts: list[str]) -> list[str]:
    """Pre-truncate long semantic documents before tokenization."""
    max_chars = model.get_max_seq_length() * _CHARS_PER_TOKEN
    return [text if len(text) <= max_chars else text[:max_chars] for text in texts]



def _encode_batch(
    model: Any,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Encode a batch of candidate semantic documents into normalized E5 embeddings."""
    _init_torch()
    import torch
    device = _get_embedding_device()
    texts = _truncate_texts_for_model(model, texts)
    texts = [PASSAGE_INSTRUCTION_PREFIX + text for text in texts]

    start = time.perf_counter()
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=device,
        )
    elapsed = time.perf_counter() - start

    if elapsed > 0:
        rate = len(texts) / elapsed
    else:
        rate = 0.0

    print(
        f"[embed_candidates] Encoded {len(texts)} docs in {elapsed:.1f}s "
        f"({rate:.1f} docs/s, batch_size={batch_size})",
        file=sys.stderr,
    )

    return embeddings.astype(np.float32)


def _classify_candidate_for_embedding(
    raw_candidate: dict[str, Any],
    total_seen: int,
    existing_ids: set[str],
) -> tuple[str, str, CandidateRetrievalDocument | None]:
    """Return a status flag, candidate_id, and document for one record."""
    candidate_id = raw_candidate.get("candidate_id", "")

    if not candidate_id:
        print(
            f"[embed_candidates] WARNING: candidate record on line {total_seen} "
            "is missing candidate_id and will be skipped.",
            file=sys.stderr,
        )
        return "missing_id", "", None

    if candidate_id in existing_ids:
        return "existing", candidate_id, None

    doc: CandidateRetrievalDocument = build_candidate_retrieval_document(raw_candidate)
    if not doc.semantic_document.strip():
        print(
            f"[embed_candidates] WARNING: candidate {candidate_id} has an "
            "empty semantic document — skipping embedding.",
            file=sys.stderr,
        )
        return "empty", candidate_id, None

    return "ready", candidate_id, doc


# ---------------------------------------------------------------------------
# 6. MAIN ENTRY POINT — generate + persist embeddings with no-recompute
# ---------------------------------------------------------------------------


def generate_candidate_embeddings(
    filtered_candidates_jsonl: str | Path,
    output_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = E5_MODEL_NAME,
    encode_chunk_size: int = 2000,
) -> None:
    """Generate and persist candidate embeddings and retrieval documents."""
    total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "candidate_embeddings.npz"
    documents_path = output_dir / "candidate_documents.jsonl"

    existing_ids: set[str] = _load_existing_embedding_ids(npz_path, required_model_name=model_name)

    print(
        f"[embed_candidates] Found {len(existing_ids):,} already-embedded "
        f"candidates. Will skip these.",
        file=sys.stderr,
    )

    model = _get_e5_model(model_name, batch_size=batch_size)

    new_candidate_ids: list[str] = []
    new_embeddings_chunks: list[np.ndarray] = []
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    chunk_records: list[dict[str, Any]] = []

    total_seen = 0
    total_skipped_existing = 0
    total_skipped_empty = 0
    total_encoded = 0

    total_candidates = _count_filtered_candidate_lines(filtered_candidates_jsonl)

    def _flush_encode_chunk(docs_fh: Any) -> None:
        nonlocal total_encoded
        if not chunk_ids:
            return

        chunk_embeddings = _encode_batch(model, chunk_texts, batch_size=batch_size)
        new_candidate_ids.extend(chunk_ids)
        new_embeddings_chunks.append(chunk_embeddings)

        for record in chunk_records:
            docs_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_encoded += len(chunk_ids)
        chunk_ids.clear()
        chunk_texts.clear()
        chunk_records.clear()

    with documents_path.open("a", encoding="utf-8") as docs_fh:
        progress = tqdm(
            _stream_filtered_candidates(filtered_candidates_jsonl),
            total=total_candidates,
            desc="embedding candidates",
            unit="cand",
            file=sys.stderr,
            leave=True,
        )
        for raw_candidate in progress:
            total_seen += 1
            status, candidate_id, doc = _classify_candidate_for_embedding(
                raw_candidate=raw_candidate,
                total_seen=total_seen,
                existing_ids=existing_ids,
            )

            if status == "missing_id":
                continue
            if status == "existing":
                total_skipped_existing += 1
                progress.set_postfix({"encoded": total_encoded, "skip": total_skipped_existing})
                continue
            if status == "empty":
                total_skipped_empty += 1
                progress.set_postfix({"encoded": total_encoded, "skip": total_skipped_existing})
                continue

            chunk_ids.append(candidate_id)
            chunk_texts.append(doc.semantic_document)
            chunk_records.append(doc.to_dict())

            if len(chunk_ids) >= encode_chunk_size:
                print(
                    f"[embed_candidates] Encoding chunk of {len(chunk_ids):,} "
                    f"(total encoded so far: {total_encoded:,})...",
                    file=sys.stderr,
                )
                _flush_encode_chunk(docs_fh)
                progress.set_postfix({"encoded": total_encoded, "skip": total_skipped_existing})

        if chunk_ids:
            print(
                f"[embed_candidates] Encoding final chunk of {len(chunk_ids):,}...",
                file=sys.stderr,
            )
            _flush_encode_chunk(docs_fh)

        progress.close()

    print(
        f"[embed_candidates] Streamed {total_seen:,} candidates: "
        f"{total_encoded:,} encoded, "
        f"{total_skipped_existing:,} already embedded (skipped), "
        f"{total_skipped_empty:,} empty documents (skipped).",
        file=sys.stderr,
    )

    if not new_candidate_ids:
        print(
            "[embed_candidates] No new candidates to embed. Nothing to do.",
            file=sys.stderr,
        )
        return

    new_embeddings = np.vstack(new_embeddings_chunks)
    existing_candidate_ids, existing_embeddings = _load_existing_embeddings(npz_path, required_model_name=model_name)

    all_candidate_ids = existing_candidate_ids + new_candidate_ids
    all_embeddings = (
        np.vstack([existing_embeddings, new_embeddings])
        if existing_embeddings.shape[0] > 0
        else new_embeddings
    )

    np.savez_compressed(
        npz_path,
        candidate_ids=np.array(all_candidate_ids, dtype="<U64"),
        embeddings=all_embeddings,
        model_name=np.array(model_name, dtype="<U64"),
    )

    print(
        f"[embed_candidates] Done. Persisted {len(all_candidate_ids):,} total "
        f"embeddings ({len(new_candidate_ids):,} new) to:\n"
        f"  {npz_path}\n"
        f"  {documents_path}",
        file=sys.stderr,
    )
    total_elapsed = time.perf_counter() - total_start
    print(f"[embed_candidates] Total embedding time: {total_elapsed:.1f}s", file=sys.stderr)


# ---------------------------------------------------------------------------
# 7. LOADING HELPERS FOR DOWNSTREAM MODULES
# ---------------------------------------------------------------------------


def load_candidate_embeddings(output_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Load persisted candidate embeddings for downstream use."""
    from retrieval.embedding.artifact_io import load_candidate_embeddings as _load

    return _load(output_dir)



def load_candidate_documents(output_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Stream persisted candidate document metadata from JSONL."""
    from retrieval.embedding.artifact_io import load_candidate_documents as _load

    yield from _load(output_dir)


# ---------------------------------------------------------------------------
# 8. CLI ENTRY POINT
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Command-line interface for embedding generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and persist E5 embeddings for filtered candidates."
    )
    parser.add_argument("input", help="Path to filtered_candidates.jsonl")
    parser.add_argument(
        "output_dir",
        help="Directory to persist candidate_embeddings.npz and candidate_documents.jsonl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"SentenceTransformer encode batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=E5_MODEL_NAME,
        help=f"E5 model identifier (default: {E5_MODEL_NAME})",
    )

    args = parser.parse_args()

    generate_candidate_embeddings(
        filtered_candidates_jsonl=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_name=args.model,
    )


if __name__ == "__main__":
    _cli()
