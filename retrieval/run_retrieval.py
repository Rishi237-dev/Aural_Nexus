"""
run_retrieval.py
================
End-to-end orchestration for the hybrid retrieval stage:

    filtered_candidates.jsonl
        → E5 embeddings + FAISS index
        → BM25 index
        → hybrid retrieval (dense + BM25 + RRF + Redrob)
        → retrieval_results.json

Run from the resume/ directory:

    python -m retrieval.run_retrieval

Or with custom paths:

    python -m retrieval.run_retrieval \\
        --candidates retrieval/filtered_candidates.jsonl \\
        --jd parsed_job_description.json \\
        --artifacts retrieval/artifacts \\
        --output retrieval/artifacts/retrieval_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# When the script is executed directly from the repository root (or any
# CWD that does not include the `resume/` folder on sys.path), Python
# cannot import the `retrieval` package. Insert the `resume/` folder
# (parent of this `retrieval/` package) onto `sys.path` early so that
# `import retrieval.*` works even when running the script from project
# root or other locations. This is safe and idempotent.
_RESUME_DIR = Path(__file__).resolve().parent.parent
if str(_RESUME_DIR) not in sys.path:
    sys.path.insert(0, str(_RESUME_DIR))

# Paths relative to resume/ (parent of retrieval/)
_RESUME_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_PATH = _RESUME_ROOT / "retrieval" / "filtered_candidates.jsonl"
DEFAULT_JD_PATH = _RESUME_ROOT / "jd" / "parsed_job_description.json"
DEFAULT_ARTIFACTS_DIR = _RESUME_ROOT / "retrieval" / "artifacts"
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACTS_DIR / "retrieval_results.json"


def _load_jd(jd_path: Path) -> dict[str, Any]:
    with jd_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_candidate_documents(candidates_jsonl: Path, artifacts_dir: Path) -> None:
    """Create candidate_documents.jsonl if it does not already exist."""
    docs_path = artifacts_dir / "candidate_documents.jsonl"
    if docs_path.exists():
        return

    print("[run_retrieval] Generating candidate_documents.jsonl (no embeddings)...", file=sys.stderr)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    from retrieval.embedding import candidate_document_builder

    with candidates_jsonl.open("r", encoding="utf-8") as inf, docs_path.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = candidate_document_builder.build_candidate_retrieval_document(raw)
            outf.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def _generate_candidate_embeddings_if_available(
    candidates_jsonl: Path,
    artifacts_dir: Path,
    lexical_only: bool,
) -> tuple[list[str], Any | None]:
    """Generate candidate embeddings when possible, otherwise fall back to none."""
    candidate_ids: list[str] = []
    embeddings = None

    if lexical_only:
        print("[run_retrieval] Lexical-only mode: skipping embedding generation and FAISS.", file=sys.stderr)
        return candidate_ids, embeddings

    try:
        from retrieval.embedding.artifact_io import load_candidate_embeddings
        from retrieval.embedding.embed_candidates import generate_candidate_embeddings
    except Exception as exc:
        print(f"[run_retrieval] embed_candidates unavailable: {exc}", file=sys.stderr)
        return candidate_ids, embeddings

    try:
        print("[run_retrieval] Step 1/5 — generating candidate embeddings...", file=sys.stderr)
        generate_candidate_embeddings(
            filtered_candidates_jsonl=candidates_jsonl,
            output_dir=artifacts_dir,
        )
    except Exception as exc:
        print(f"[run_retrieval] Skipping embedding generation: {exc}", file=sys.stderr)

    try:
        candidate_ids, embeddings = load_candidate_embeddings(artifacts_dir)
    except Exception as exc:
        print(f"[run_retrieval] No usable embedding archive: {exc}", file=sys.stderr)

    return candidate_ids, embeddings


def _load_or_build_faiss_index(
    artifacts_dir: Path,
    embeddings: Any | None,
    force_rebuild_indexes: bool,
) -> Any | None:
    """Load or build FAISS only when embeddings are available."""
    faiss_index_path = artifacts_dir / "candidate.index"
    if embeddings is None:
        print("[run_retrieval] No embeddings available — skipping FAISS/dense retrieval.", file=sys.stderr)
        return None

    try:
        from retrieval.embedding.build_faiss import build_and_persist_faiss_index, load_faiss_index

        if force_rebuild_indexes or not faiss_index_path.exists():
            print("[run_retrieval] Step 2/5 — building FAISS index...", file=sys.stderr)
            build_and_persist_faiss_index(artifacts_dir, faiss_index_path)
        else:
            print("[run_retrieval] Step 2/5 — FAISS index exists, loading...", file=sys.stderr)
        return load_faiss_index(faiss_index_path)
    except Exception as exc:
        print(f"[run_retrieval] FAISS unavailable or failed: {exc}", file=sys.stderr)
        return None


def _load_or_build_bm25_index(artifacts_dir: Path, force_rebuild_indexes: bool) -> Any | None:
    """Load or build BM25, falling back to a simple lexical scorer when needed."""
    bm25_index_path = artifacts_dir / "candidate_bm25.pkl"
    try:
        from retrieval.embedding.build_bm25 import build_and_persist_bm25_index, load_bm25_index

        if force_rebuild_indexes or not bm25_index_path.exists():
            print("[run_retrieval] Step 3/5 — building BM25 index...", file=sys.stderr)
            build_and_persist_bm25_index(artifacts_dir, bm25_index_path)
        else:
            print("[run_retrieval] Step 3/5 — BM25 index exists, loading...", file=sys.stderr)

        return load_bm25_index(bm25_index_path)
    except Exception as exc:
        print(f"[run_retrieval] rank_bm25 or build_bm25 unavailable: {exc}", file=sys.stderr)

    from collections import defaultdict
    import math

    docs_path = artifacts_dir / "candidate_documents.jsonl"
    candidate_ids_lex: list[str] = []
    lexical_docs: list[str] = []
    if docs_path.exists():
        with docs_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                candidate_ids_lex.append(rec["candidate_id"])
                lexical_docs.append(rec.get("lexical_document", ""))

    df = defaultdict(int)
    tokenized_corpus = []
    for doc in lexical_docs:
        tokens = set(doc.split()) if doc else set()
        tokenized_corpus.append(tokens)
        for token in tokens:
            df[token] += 1

    total_docs = max(1, len(tokenized_corpus))

    def simple_search(query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        scores = []
        query_terms = set(query_tokens)
        for cid, tokens in zip(candidate_ids_lex, tokenized_corpus):
            overlap = query_terms & tokens
            if not overlap:
                continue
            score = 0.0
            for token in overlap:
                score += math.log((total_docs + 1) / (1 + df.get(token, 0)))
            scores.append((cid, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    class SimpleLexicalIndex:
        def __init__(self, search_fn):
            self.search = search_fn

    return SimpleLexicalIndex(simple_search)


def _load_jd_query_embedding(faiss_index: Any | None, jd_doc: Any) -> Any | None:
    """Load the JD embedding only when dense retrieval is active."""
    if faiss_index is None:
        return None

    try:
        from retrieval.embedding.embed_candidates import _get_e5_model
        from retrieval.embedding.hybrid_retriever import embed_jd_query

        print("[run_retrieval] Step 4/5 — encoding JD query embedding...", file=sys.stderr)
        model = _get_e5_model()
        return embed_jd_query(jd_doc.semantic_document, model)
    except Exception as exc:
        print(f"[run_retrieval] Skipping JD embedding: {exc}", file=sys.stderr)
        return None


def _run_retrieval(
    faiss_index: Any | None,
    candidate_ids: list[str],
    bm25_index: Any | None,
    jd_query_embedding: Any | None,
    jd_doc: Any,
    redrob_signals_map: dict[str, dict[str, Any]],
    per_method_top_k: int,
    output_top_k: int,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval if possible, otherwise lexical fallback."""
    try:
        from retrieval.embedding.hybrid_retriever import run_hybrid_retrieval
    except Exception as exc:
        print(f"[run_retrieval] hybrid_retriever unavailable: {exc}", file=sys.stderr)
        run_hybrid_retrieval = None

    print(
        "[run_retrieval] Step 5/5 — running retrieval (hybrid if available, else lexical-only)...",
        file=sys.stderr,
    )

    if (
        run_hybrid_retrieval is not None
        and faiss_index is not None
        and jd_query_embedding is not None
        and bm25_index is not None
    ):
        results = run_hybrid_retrieval(
            faiss_index=faiss_index,
            faiss_candidate_ids=candidate_ids,
            bm25_index=bm25_index,
            jd_query_embedding=jd_query_embedding,
            jd_lexical_document=jd_doc.lexical_document,
            candidate_redrob_signals=redrob_signals_map,
            per_method_top_k=per_method_top_k,
            output_top_k=output_top_k,
        )
        return [result.to_dict() for result in results]

    jd_tokens = jd_doc.lexical_document.split()
    bm25_results: list[tuple[str, float]] = []
    if bm25_index is not None:
        bm25_results = bm25_index.search(jd_tokens, top_k=output_top_k)

    from retrieval.embedding.redrob_adjustment import compute_redrob_multiplier

    results: list[dict[str, Any]] = []
    for cid, score in bm25_results[:output_top_k]:
        multiplier = compute_redrob_multiplier(redrob_signals_map.get(cid, {}))
        results.append(
            {
                "candidate_id": cid,
                "dense_score": 0.0,
                "bm25_score": score,
                "rrf_score": 0.0,
                "redrob_multiplier": multiplier,
                "final_retrieval_score": score * multiplier,
            }
        )
    return results


def _load_redrob_signals_map(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    """Build candidate_id → redrob_signals from persisted document metadata."""
    from retrieval.embedding.artifact_io import load_candidate_documents

    signals_map: dict[str, dict[str, Any]] = {}
    for record in load_candidate_documents(artifacts_dir):
        signals_map[record["candidate_id"]] = record.get("redrob_signals", {})
    return signals_map


def run_retrieval_pipeline(
    candidates_jsonl: str | Path = DEFAULT_CANDIDATES_PATH,
    jd_path: str | Path = DEFAULT_JD_PATH,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    per_method_top_k: int = 6000,
    output_top_k: int = 3000,
    force_rebuild_indexes: bool = False,
    lexical_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the full retrieval pipeline and write ranked results to disk.

    Returns the list of RetrievalResult dicts (top output_top_k).
    """
    candidates_jsonl = Path(candidates_jsonl)
    jd_path = Path(jd_path)
    artifacts_dir = Path(artifacts_dir)
    output_path = Path(output_path)

    if not candidates_jsonl.exists():
        raise FileNotFoundError(
            f"Filtered candidates file not found: {candidates_jsonl}\n"
            "Run retrieval/filtering.py first (after my_task.py gate)."
        )
    if not jd_path.exists():
        raise FileNotFoundError(f"Parsed JD not found: {jd_path}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    from retrieval.embedding.jd_document_builder import build_jd_retrieval_document

    jd_dict = _load_jd(jd_path)
    jd_doc = build_jd_retrieval_document(jd_dict)

    print("[run_retrieval] JD semantic document length:", len(jd_doc.semantic_document), file=sys.stderr)
    print("[run_retrieval] JD lexical token count:", len(jd_doc.lexical_document.split()), file=sys.stderr)

    _ensure_candidate_documents(candidates_jsonl, artifacts_dir)

    candidate_ids, embeddings = _generate_candidate_embeddings_if_available(candidates_jsonl, artifacts_dir, lexical_only)
    faiss_index = _load_or_build_faiss_index(artifacts_dir, embeddings, force_rebuild_indexes)
    bm25_index = _load_or_build_bm25_index(artifacts_dir, force_rebuild_indexes)
    jd_query_embedding = _load_jd_query_embedding(faiss_index, jd_doc)

    print("[run_retrieval] Loading redrob signals from document metadata...", file=sys.stderr)
    redrob_signals_map = _load_redrob_signals_map(artifacts_dir)

    results = _run_retrieval(
        faiss_index=faiss_index,
        candidate_ids=candidate_ids,
        bm25_index=bm25_index,
        jd_query_embedding=jd_query_embedding,
        jd_doc=jd_doc,
        redrob_signals_map=redrob_signals_map,
        per_method_top_k=per_method_top_k,
        output_top_k=output_top_k,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(
        f"[run_retrieval] Done — {len(results):,} candidates written to {output_path}",
        file=sys.stderr,
    )
    if results:
        print(
            f"[run_retrieval] Top candidate: {results[0]['candidate_id']} "
            f"(score={results[0]['final_retrieval_score']:.6f})",
            file=sys.stderr,
        )

    return results


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run hybrid candidate retrieval (E5 + FAISS + BM25 + RRF)."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help="Filtered candidates JSONL (~85k gate survivors).",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=DEFAULT_JD_PATH,
        help="Parsed job description JSON.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for embeddings, indexes, and intermediate files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path for ranked retrieval results.",
    )
    parser.add_argument(
        "--per-method-top-k",
        type=int,
        default=6000,
        help="Top-K per retrieval method before RRF fusion (default: 6000).",
    )
    parser.add_argument(
        "--output-top-k",
        type=int,
        default=3000,
        help="Final number of candidates to return (default: 3000).",
    )
    parser.add_argument(
        "--force-rebuild-indexes",
        action="store_true",
        help="Rebuild FAISS and BM25 indexes even if they already exist.",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Run lexical-only retrieval (skip embeddings and FAISS).",
    )

    args = parser.parse_args()

    run_retrieval_pipeline(
        candidates_jsonl=args.candidates,
        jd_path=args.jd,
        artifacts_dir=args.artifacts,
        output_path=args.output,
        per_method_top_k=args.per_method_top_k,
        output_top_k=args.output_top_k,
        force_rebuild_indexes=args.force_rebuild_indexes,
        lexical_only=args.lexical_only,
    )


if __name__ == "__main__":
    _cli()
