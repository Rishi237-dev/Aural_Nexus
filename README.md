# JBF â€” Job-Fit Benchmark

A deterministic candidate ranking pipeline for AI engineering and retrieval roles. Combines a keyword-evidence scoring model with a full **hybrid dense + lexical retrieval system** (E5 embeddings, FAISS, BM25, RRF) to rank candidates against a structured job description. Produces a ranked shortlist with per-candidate scores, strengths, weaknesses, and full diagnostic output files.

---

## Objective

Given a dataset of candidate profiles in JSONL format, rank every candidate against a target job description â€” currently configured for a **Senior AI Engineer** role at **Redrob AI** (talent intelligence platform) covering Search, Retrieval, Ranking, NLP, and AI Engineering domains. Produce a ranked shortlist with per-candidate scores, explanations, and a spec-compliant submission CSV.

Every point in a candidate's score maps back to a specific, named signal â€” the pipeline is fully interpretable.

---

## Implemented Pipeline

```
job_description.txt
      â”‚
      â–¼
  [ JD Parser ]               jd/jd_parser.py
  Parses raw JD text into a structured JDSchema
  (regex + heuristic-based, stdlib only)
  Outputs â†’ jd/parsed_job_description.json
      â”‚
      â–¼
  [ Retrieval Pipeline ]      retrieval/run_retrieval.py
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚  Candidate Document Builder                         â”‚
  â”‚    â†’ JSONL profiles â†’ semantic + lexical docs       â”‚
  â”‚  Embedding Generator  (E5-base-v2, cuda:0 / CPU)   â”‚
  â”‚    â†’ candidate_embeddings.npz  (85k candidates)     â”‚
  â”‚  FAISS Index Builder                                â”‚
  â”‚    â†’ candidate.index  (IndexFlatIP, 768-dim)        â”‚
  â”‚  BM25 Index Builder   (rank_bm25, BM25Okapi)        â”‚
  â”‚    â†’ candidate_bm25.pkl                             â”‚
  â”‚  JD Document Builder + JD Query Embedding           â”‚
  â”‚    â†’ E5 query vector on cuda:0 / CPU                â”‚
  â”‚  Hybrid Retriever                                   â”‚
  â”‚    â†’ Dense FAISS search (top-6,000)                 â”‚
  â”‚    + BM25 lexical search (top-6,000)                â”‚
  â”‚    â†’ RRF fusion â†’ 10,000+ unique candidates         â”‚
  â”‚    â†’ Redrob Adjustment (availability multiplier)    â”‚
  â”‚  Output â†’ retrieval/artifacts/retrieval_results.jsonâ”‚
  â”‚           (top-3,000 with final_retrieval_score)    â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
      â”‚
      â–¼
candidates.jsonl
      â”‚
      â–¼
  [ Parser ]                  candidate/parser.py
  Loads and normalises raw JSONL into typed Candidate dataclasses
      â”‚
      â–¼
  [ JD Feature Extractor ]    features/jd_feature_extractor.py
  Counts keyword matches across 7 JD categories in the full profile text
      â”‚
      â–¼
  [ Evidence Locator ]        features/evidence_locator.py
  Finds which profile sections (headline / summary / skills / career)
  contain each keyword category â€” called exactly once per candidate
      â”‚
      â–¼
  [ Evidence Engine ]         features/evidence_engine.py
  Computes a weighted evidence-strength score (0â€“100) per category
  and in aggregate, honouring career > summary > headline > skills
      â”‚
      â–¼
  [ Evidence Quality ]        features/evidence_quality.py
  Produces a credibility multiplier [0.5, 1.0] â€” penalises profiles
  where skills claims are not backed by any career history
      â”‚
      â–¼
  [ Consistency Engine ]      risk/consistency_engine.py
  Assesses internal coherence: title â†” career, summary â†” career,
  skills spread, career progression plausibility, timeline realism
      â”‚
      â–¼
  [ Feature Vector ]          features/feature_vector.py
  Assembles all upstream outputs into a single flat dict â€”
  no module below this layer repeats any text scan
      â”‚
      â–¼
  [ Signal Fusion ]           ranking/signal_fusion.py
  Computes the final score as a pure additive weighted sum (0â€“100),
  integrating all deterministic signals + the retrieval_ext component.
  Generates human-readable strengths and weaknesses.
      â”‚
      â–¼
  [ Output Layer ]            utils/pipeline_output.py
  Writes diagnostics.json, run_summary.json, ranked_candidates.csv,
  and pipeline.log
      â”‚
      â–¼
  [ Reasoning Generator ]     ranking/reasoning_generator.py
  Produces a 1-2 sentence, deterministic, rank-aware reasoning
  string per candidate â€” cites real profile values, no hallucination
      â”‚
      â–¼
  [ Submission Generator ]    utils/submission_generator.py
  Formats top 100 into a spec-compliant CSV (candidate_id, rank,
  score, reasoning) with tie-breaking and monotonicity enforcement
      â”‚
      â–¼
  outputs/submission.csv      Validated by validate_submission.py
```

---

## Key Features

### Retrieval Pipeline â€” Hybrid Dense + Lexical (v2.0.0)

The retrieval module (`retrieval/`) generates a semantic + lexical relevance score for every candidate against the JD, independently of the keyword-matching pipeline. Its output is fused into the final score as the `retrieval_ext` component.

#### Architecture

| Stage | Module | Technology |
|---|---|---|
| Candidate document building | `candidate_document_builder.py` | Pure Python |
| JD document building | `jd_document_builder.py` | Pure Python |
| Candidate embedding | `embed_candidates.py` | E5-base-v2 via sentence-transformers |
| JD query embedding | `hybrid_retriever.py` | E5-base-v2 (query prefix) |
| Dense index | `build_faiss.py` | FAISS IndexFlatIP (768-dim) |
| Lexical index | `build_bm25.py` | rank_bm25 BM25Okapi |
| Hybrid fusion | `hybrid_retriever.py` | Reciprocal Rank Fusion (RRF, k=60) |
| Availability adjustment | `redrob_adjustment.py` | Bounded multiplier [0.85, 1.05] |
| Artifact I/O | `artifact_io.py` | NumPy .npz + pickle |

#### GPU Device Selection

Embedding generation automatically uses **`cuda:0`** when CUDA is available (e.g., NVIDIA RTX 3050 Laptop GPU), with a transparent CPU fallback. All other pipeline operations (BM25, FAISS indexing, document building, orchestration) remain CPU-based.

A startup banner is printed before model load:

```
[embed_candidates] â”€â”€â”€ Device Selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
[embed_candidates]   Selected device : cuda:0
[embed_candidates]   GPU name        : NVIDIA GeForce RTX 3050 6GB Laptop GPU
[embed_candidates]   CUDA available  : True
[embed_candidates]   Batch size      : 192
[embed_candidates] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

#### No-Recompute Guarantees

- Candidate embeddings already in `candidate_embeddings.npz` are never re-encoded
- FAISS and BM25 indexes are reused on subsequent runs unless `--force-rebuild-indexes` is passed
- Only new candidates (if any) trigger incremental re-embedding

#### Retrieval Output

`retrieval/artifacts/retrieval_results.json` â€” top-3,000 candidates ranked by `final_retrieval_score`:

```json
[
  {
    "candidate_id": "CAND_0081846",
    "dense_score": 0.927631,
    "bm25_score": 491.12,
    "rrf_score": 0.031746,
    "redrob_multiplier": 1.02,
    "final_retrieval_score": 0.032371
  }
]
```

#### Retrieval Artifacts

| File | Size | Description |
|---|---|---|
| `candidate_embeddings.npz` | ~144 MB | E5 embeddings for all 85,187 filtered candidates |
| `candidate.index` | ~250 MB | FAISS IndexFlatIP (768-dim) |
| `candidate_bm25.pkl` | ~114 MB | BM25Okapi index |
| `candidate_documents.jsonl` | ~334 MB | Semantic + lexical document texts per candidate |
| `retrieval_results.json` | ~0.6 MB | Final ranked retrieval scores (top-3,000) |

---

### JD Parser â€” Structured Job Description Understanding

The JD parser (`jd/jd_parser.py`) converts raw job description text into a fully structured JSON representation. It is the **single source of truth** for all downstream modules.

**Capabilities:**
- Section detection for conventional and unconventional headings
- Structured extraction: job title, company, location, employment type, experience range, required/preferred skills, technologies, responsibilities, behavioral expectations, negative requirements, evaluation metrics
- Domain inference across 11 categories
- Technology vocabulary of 80+ named tools
- Concept keyword extraction and hyphen-normalised deduplication

**Output schema** (`jd/parsed_job_description.json`):
```json
{
  "job_title": "Senior AI Engineer â€” Founding Team",
  "company": "Redrob AI (Series A AI-native talent intelligence platform)",
  "location": "Pune/Noida, India (Hybrid â€” flexible cadence)",
  "employment_type": ["Full-time", "Hybrid"],
  "domain": ["Search", "Information Retrieval", "NLP", "Ranking Systems"],
  "experience_requirements": { "min_years": 5, "max_years": 9 },
  "required_skills": ["..."],
  "required_technologies": ["sentence-transformers", "elasticsearch", "faiss"],
  "evaluation_metrics": ["NDCG", "MRR", "MAP", "A/B TESTING"],
  "keywords": ["embeddings", "retrieval", "ranking", "vector databases"]
}
```

---

### Signal Fusion â€” Seven-Component Additive Score

The final score is a **pure weighted sum** of seven normalised components. There are no multipliers that collapse scores.

| Component | Max pts | Signal source |
|---|---|---|
| Retrieval experience | 25 | Keyword evidence (career-backed) |
| Ranking / evaluation | 18 | Keyword evidence (career-backed) |
| Vector DB experience | 10 | Keyword evidence (career-backed) |
| Years of experience | 12 | Structured profile field |
| Profile consistency | 13 | Five-dimension consistency engine |
| Evidence credibility | 7 | Credibility multiplier [0.5, 1.0] |
| **Retrieval ext** | **15** | **Hybrid retrieval score (FAISS + BM25 + RRF)** |
| **Total** | **100** | |

#### Dynamic `retrieval_ext` Normalisation

The raw `final_retrieval_score` is normalised to [0, 100] using the actual min/max values computed at runtime from `retrieval_results.json` â€” no hardcoded constants:

```python
# main.py â€” computed from the live retrieval output
retrieval_ext_min = min(all_scores)
retrieval_ext_max = max(all_scores)

# signal_fusion.py â€” passed in as parameters
normalised = (raw - score_min) / (score_max - score_min) * 100
```

Candidates not in the retrieval top-3,000 receive `retrieval_score = 0.0` (default lookup miss).

---

### Evidence Localisation

Keywords are matched separately against four profile sections: **headline**, **summary**, **skills list**, and **career descriptions**. Career-backed evidence is weighted 4Ã— vs. a skills-list mention.

### Credibility Assessment

Three penalty patterns are detected:
- Non-technical title with dense AI skill claims and zero career backing (âˆ’0.30)
- 8+ AI skill keywords with zero career backing â€” "course collector" pattern (âˆ’0.20)
- 4â€“7 AI skill keywords with zero career backing â€” moderate concern (âˆ’0.10)

Credibility floor is 0.5.

### Consistency Engine (Five Sub-scores)

| Sub-score | What it measures |
|---|---|
| Title alignment | Does the title match the career evidence? |
| Summary alignment | Does the summary claim things career history backs up? |
| Skill alignment | Are claimed skills evidenced across multiple sections? |
| Career progression | Is the seniority level plausible for the years claimed? |
| Timeline plausibility | Is the claimed level reachable in the stated timeframe? |

### Reasoning Generator â€” Deterministic, Rank-Aware Explanations

Produces a 1â€“2 sentence explanation per top-100 candidate. Reads only from `calculate_final_score()` results â€” invents nothing.

| Band | Ranks | Tone |
|---|---|---|
| Top | 1â€“10 | Lead with strengths, mention concern only if material |
| Strong | 11â€“40 | Lead with strengths, note one concern |
| Moderate | 41â€“70 | Balanced: one strength, one concern |
| Marginal | 71â€“100 | Lead with the gap, strength only if real |

### Submission Generator â€” Spec-Compliant Competition CSV

Produces the official top-100 submission CSV with:
- Monotonicity enforcement (raises `ValueError` on violation)
- Tie-breaking applied before top-100 slice
- `candidate_id` validation against `CAND_[0-9]{7}`

### Audit Modules (Optional, Flag-Gated)

- **Dataset audit** (`RUN_AUDIT = True`) â†’ `outputs/audit_report.txt`
- **JD skill audit** (`RUN_JD_AUDIT = True`) â†’ `outputs/jd_skill_audit.txt`

---

## Project Structure

```
JBF_Test/
â”‚
â”œâ”€â”€ main.py                             # Orchestration entry point â€” no business logic
â”‚
â”œâ”€â”€ jd/
â”‚   â”œâ”€â”€ jd_parser.py                    # JD text â†’ structured JDSchema
â”‚   â”œâ”€â”€ job_description.txt             # Raw JD input (Redrob AI Senior AI Engineer)
â”‚   â””â”€â”€ parsed_job_description.json     # Structured output (auto-generated)
â”‚
â”œâ”€â”€ retrieval/                          # Hybrid retrieval pipeline
â”‚   â”œâ”€â”€ run_retrieval.py                # Entry point â€” orchestrates all retrieval steps
â”‚   â”œâ”€â”€ filtering.py                    # Candidate pre-filtering logic
â”‚   â”œâ”€â”€ filtered_candidates.jsonl       # Pre-filtered candidate pool (~85k candidates)
â”‚   â”œâ”€â”€ archi.md                        # Retrieval architecture specification
â”‚   â”œâ”€â”€ artifacts/
â”‚   â”‚   â”œâ”€â”€ candidate_embeddings.npz    # E5 embeddings (85k Ã— 768, float32)
â”‚   â”‚   â”œâ”€â”€ candidate.index             # FAISS IndexFlatIP
â”‚   â”‚   â”œâ”€â”€ candidate_bm25.pkl          # BM25Okapi index
â”‚   â”‚   â”œâ”€â”€ candidate_documents.jsonl   # Semantic + lexical docs per candidate
â”‚   â”‚   â””â”€â”€ retrieval_results.json      # Final ranked scores (top-3,000)
â”‚   â””â”€â”€ embedding/
â”‚       â”œâ”€â”€ embed_candidates.py         # E5 embedding generation (GPU/CPU auto-select)
â”‚       â”œâ”€â”€ hybrid_retriever.py         # Dense + BM25 + RRF + Redrob fusion
â”‚       â”œâ”€â”€ build_faiss.py              # FAISS index build and load
â”‚       â”œâ”€â”€ build_bm25.py               # BM25Okapi index build and load
â”‚       â”œâ”€â”€ candidate_document_builder.py  # Profile â†’ semantic+lexical documents
â”‚       â”œâ”€â”€ jd_document_builder.py      # JD â†’ semantic+lexical documents
â”‚       â”œâ”€â”€ redrob_adjustment.py        # Availability signal multiplier [0.85, 1.05]
â”‚       â””â”€â”€ artifact_io.py              # NumPy .npz and pickle I/O helpers
â”‚
â”œâ”€â”€ candidate/
â”‚   â”œâ”€â”€ schema.py                       # Candidate dataclass definition
â”‚   â””â”€â”€ parser.py                       # JSONL loader â†’ Candidate objects
â”‚
â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ jd_feature_extractor.py         # Keyword match counts across 7 JD categories
â”‚   â”œâ”€â”€ evidence_locator.py             # Per-section keyword evidence dict
â”‚   â”œâ”€â”€ evidence_engine.py              # Weighted evidence-strength score (0â€“100)
â”‚   â”œâ”€â”€ evidence_quality.py             # Credibility multiplier [0.5, 1.0]
â”‚   â””â”€â”€ feature_vector.py               # Assembles all upstream outputs into one flat dict
â”‚
â”œâ”€â”€ risk/
â”‚   â””â”€â”€ consistency_engine.py           # Five-dimension internal consistency score
â”‚
â”œâ”€â”€ ranking/
â”‚   â”œâ”€â”€ signal_fusion.py                # Seven-component additive weighted score
â”‚   â””â”€â”€ reasoning_generator.py          # Deterministic 1-2 sentence reasoning
â”‚
â”œâ”€â”€ audit/
â”‚   â”œâ”€â”€ dataset_audit.py                # Dataset-wide statistics report
â”‚   â””â”€â”€ jd_skill_audit.py               # JD keyword frequency across all candidates
â”‚
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ pipeline_output.py              # Logging setup + all output file writers
â”‚   â””â”€â”€ submission_generator.py         # Official top-100 submission CSV generator
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ candidates.jsonl                # Input dataset (100,000 candidates)
â”‚
â”œâ”€â”€ outputs/                            # All generated files (overwritten on every run)
â”‚   â”œâ”€â”€ pipeline.log
â”‚   â”œâ”€â”€ diagnostics.json
â”‚   â”œâ”€â”€ run_summary.json
â”‚   â”œâ”€â”€ ranked_candidates.csv
â”‚   â”œâ”€â”€ submission.csv
â”‚   â”œâ”€â”€ audit_report.txt               # Only present if RUN_AUDIT = True
â”‚   â””â”€â”€ jd_skill_audit.txt             # Only present if RUN_JD_AUDIT = True
â”‚
â”œâ”€â”€ validate_submission.py              # Official submission validator
â””â”€â”€ requirements.txt
```

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback is automatic)

### Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.11.0+cu128 | GPU-accelerated embedding inference |
| `sentence-transformers` | 5.6.0 | E5-base-v2 model wrapper |
| `faiss-cpu` | 1.14.3 | FAISS dense index |
| `rank-bm25` | 0.2.2 | BM25 lexical index |
| `numpy` | 2.5.0 | Embedding matrix storage |
| `tqdm` | 4.68.3 | Progress bars |
| `scikit-learn` | 1.9.0 | Supporting utilities |

### Prepare data

Place your candidate dataset at `data/candidates.jsonl`. Each line must be a JSON object:

```json
{
  "candidate_id": "CAND_0000001",
  "profile": {
    "headline": "...",
    "summary": "...",
    "years_of_experience": 5,
    "current_title": "...",
    "current_company": "...",
    "country": "..."
  },
  "skills": [{ "name": "Python" }, { "name": "FAISS" }],
  "career_history": [{ "title": "...", "description": "...", "industry": "..." }],
  "education": [],
  "redrob_signals": {
    "open_to_work_flag": true,
    "recruiter_response_rate": 0.72,
    "github_activity_score": 85,
    "notice_period_days": 30
  }
}
```

---

## How to Run

### Run the complete pipeline

```bash
python main.py
```

This single command runs the entire end-to-end pipeline:

1. Parses the job description â†’ `jd/parsed_job_description.json`
2. Runs the retrieval pipeline (reuses cached embeddings/indexes when available) â†’ `retrieval/artifacts/retrieval_results.json`
3. Loads all 100,000 candidates and scores every one against the JD
4. Writes all output files â†’ `outputs/`
5. Generates the spec-compliant submission â†’ `outputs/submission.csv`

**No flags or arguments required.** GPU is used automatically when available.

### Validate the submission (optional)

```bash
python validate_submission.py outputs/submission.csv
```

Expected output: `Submission is valid.`

### Run the retrieval pipeline independently

```bash
python -m retrieval.run_retrieval
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--jd PATH` | `jd/parsed_job_description.json` | Override JD path |
| `--candidates PATH` | `retrieval/filtered_candidates.jsonl` | Override candidate JSONL |
| `--artifacts-dir PATH` | `retrieval/artifacts/` | Override artifacts directory |
| `--output PATH` | `retrieval/artifacts/retrieval_results.json` | Override output path |
| `--force-rebuild-indexes` | off | Force rebuild of FAISS and BM25 indexes |
| `--lexical-only` | off | Skip dense retrieval, use BM25 only |

### Configuration (top of `main.py`)

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `"data/candidates.jsonl"` | Path to input dataset |
| `RUN_AUDIT` | `False` | Run dataset-wide statistics audit |
| `RUN_JD_AUDIT` | `False` | Run JD keyword frequency audit |
| `TOP_N` | `20` | Number of top candidates printed to console |

---

## Output Files

All files are written to `outputs/` and **overwritten on every run**.

### `pipeline.log`

```
2026-07-01 18:48:14  INFO  Pipeline started.
2026-07-01 18:48:14  INFO  Parsing Job Description â†’ jd/parsed_job_description.json
2026-07-01 18:48:14  INFO  Retrieval pipeline started.
2026-07-01 18:49:38  INFO  Retrieval pipeline completed.
2026-07-01 18:49:38  INFO  Dataset loaded: data/candidates.jsonl â€” 100000 candidates.
2026-07-01 18:49:38  INFO  Ranking started â€” processing 100000 candidates.
2026-07-01 18:49:46  INFO  Ranking completed â€” processed: 100000  skipped: 0.
2026-07-01 18:49:46  INFO  Submission written â€” 100 rows -> outputs/submission.csv
2026-07-01 18:49:46  INFO  Pipeline finished â€” 45.19s  (100000 candidates).
```

### `diagnostics.json`

```json
{
  "candidate_count": 100000,
  "processed_candidates": 100000,
  "skipped_candidates": 0,
  "highest_score": 98.393,
  "lowest_score": 10.61,
  "average_score": 29.12,
  "median_score": 28.18,
  "average_consistency": 41.99,
  "average_credibility": 0.9884,
  "average_years_experience": 7.17,
  "candidates_with_retrieval_evidence": 12473,
  "candidates_with_ranking_evidence": 3429,
  "candidates_with_vector_db_evidence": 13816
}
```

### `run_summary.json`

```json
{
  "timestamp": "2026-07-01T13:19:46+00:00",
  "dataset": "data/candidates.jsonl",
  "candidate_count": 100000,
  "processed": 100000,
  "skipped": 0,
  "total_runtime_seconds": 44.58,
  "highest_score": 98.393,
  "average_score": 29.12
}
```

### `ranked_candidates.csv`

All 100,000 candidates sorted by final score descending:

```
rank,candidate_id,current_title,years_experience,final_score
1,CAND_0039754,Senior Applied Scientist,16.2,98.39
2,CAND_0046064,Senior NLP Engineer,8.9,98.25
```

### `submission.csv`

Official top-100 submission CSV:

```
candidate_id,rank,score,reasoning
CAND_0039754,1,98.3930,"16.2-year Senior Applied Scientist with strong retrieval..."
CAND_0046064,2,98.2500,"Senior NLP Engineer with 8.9 years..."
```

Validate with: `python validate_submission.py outputs/submission.csv`

---

## Performance

Measured on 100,000 candidates (NVIDIA RTX 3050 6GB Laptop GPU, cached embeddings + indexes):

| Stage | Time |
|---|---|
| JD parsing | < 1 s |
| Retrieval pipeline (cached embeddings + indexes) | ~29 s |
| Ranking loop (100k candidates, 7 components) | ~14 s |
| Output writing + submission generation | < 1 s |
| **Total (`python main.py`)** | **~45 s** |

> On the **first run** (no cached artifacts), the retrieval pipeline generates embeddings for ~85,000 candidates on the GPU before building indexes. Subsequent runs skip embedding and go straight to FAISS + BM25 search.

---

## Technologies Used

| Technology | Role |
|---|---|
| Python 3.9+ | Core language |
| `torch` (CUDA) | GPU-accelerated embedding inference |
| `sentence-transformers` | E5-base-v2 embedding model |
| `faiss-cpu` | Dense vector index (FAISS IndexFlatIP) |
| `rank-bm25` | Lexical BM25Okapi index |
| `numpy` | Embedding matrix + artifact storage |
| `tqdm` | Retrieval progress bars |
| `dataclasses` | Typed schemas â€” Candidate, JDSchema (stdlib) |
| `re` | JD parsing and keyword extraction (stdlib) |
| `logging` | Structured pipeline logging (stdlib) |
| `csv` / `json` | Output file generation (stdlib) |

---

## Current Limitations

- **Keyword-based ranking signal.** The scoring model uses substring keyword matching. It cannot understand context.
- **Retrieval covers ~85k pre-filtered candidates.** Candidates outside the filtered pool receive `retrieval_ext = 0.0`.
- **Retrieval search re-runs every invocation.** FAISS + BM25 search re-runs on every `python main.py`. This is fast (~29 s with cached indexes) but not skipped when the JD has not changed.
- **JD parser is heuristic-based.** Highly unusual JD formats may require adding new patterns.
- **Single JD target.** Keyword lists and weights are calibrated for one specific role.
- **Single-threaded ranking loop.** The 100k candidate scoring loop is sequential.
- **No API or web interface.** CLI only.

---

## Contributing

1. Do not modify scoring logic in `evidence_engine.py`, `evidence_quality.py`, `consistency_engine.py`, or `signal_fusion.py` without a corresponding design decision.
2. All new output concerns belong in `utils/pipeline_output.py`.
3. `main.py` must remain an orchestration-only file â€” no business logic.
4. JD parsing changes belong exclusively in `jd/jd_parser.py`. No other module reads the raw JD text.
5. Do not modify modules outside the `retrieval/` package from within the retrieval pipeline â€” it is a self-contained subsystem.
6. `reasoning_generator.py` must never invent facts â€” every claim must trace to a value in the `calculate_final_score()` result dict.
7. `submission_generator.py` must never silently produce an invalid CSV â€” monotonicity violations raise `ValueError` immediately.
8. Run `python validate_submission.py outputs/submission.csv` after any change to the reasoning or submission modules.
9. Run the full pipeline after any change and verify `outputs/diagnostics.json` is consistent before merging.

---

## Changelog

### v2.0.0 â€” Hybrid Retrieval Pipeline Integration

- **New package**: `retrieval/` â€” complete hybrid retrieval system integrated into `python main.py`
  - `run_retrieval.py` â€” 5-step orchestrator with graceful fallbacks at every stage
  - `embed_candidates.py` â€” E5-base-v2 embedding generation; automatic `cuda:0` / CPU selection; FP16 GPU inference; no-recompute guarantee (skips already-embedded candidates); startup device banner logging device name, CUDA status, and batch size
  - `build_faiss.py` â€” FAISS IndexFlatIP build, persist, and load
  - `build_bm25.py` â€” BM25Okapi index build, persist (pickle), and load
  - `hybrid_retriever.py` â€” dense FAISS search (top-6k) + BM25 search (top-6k) + RRF fusion (k=60) + Redrob availability adjustment; JD query encoded on same device as candidates
  - `candidate_document_builder.py` â€” profile â†’ semantic + lexical document pair
  - `jd_document_builder.py` â€” JD JSON â†’ semantic + lexical document pair
  - `redrob_adjustment.py` â€” bounded availability multiplier [0.85, 1.05] from open-to-work, recruiter response, GitHub activity, notice period
  - `artifact_io.py` â€” safe NumPy .npz and pickle I/O with model-name validation
- **Updated**: `signal_fusion.py` â€” added `retrieval_ext` as a 7th scoring component; accepts dynamic `score_min`/`score_max` bounds; re-balanced weights to 100-point total; updated strengths/weaknesses generation for retrieval signal
- **Updated**: `main.py` â€” wired `run_retrieval_pipeline()` as a mandatory pipeline step between JD parsing and candidate ranking; retrieval score lookup built dynamically from live output; dynamic min/max normalisation bounds computed at runtime from the live file
- **Fixed**: `retrieval/run_retrieval.py` default JD path corrected from project root to `jd/parsed_job_description.json`
- **Performance**: End-to-end `python main.py` runtime ~45 s on 100k candidates with GPU + cached embeddings/indexes

### v1.2.0 â€” Reasoning Generator & Submission Generator

- **New module**: `ranking/reasoning_generator.py` â€” deterministic, rank-aware 1â€“2 sentence reasoning per candidate; dispatch-table architecture; four rank bands; no randomness
- **New module**: `utils/submission_generator.py` â€” spec-compliant top-100 competition CSV; monotonicity violation raises `ValueError`; tie-breaking applied before slice
- **New file**: `validate_submission.py` â€” official competition validator
- **Updated**: `main.py` â€” submission generation integrated as a final pipeline step

### v1.1.0 â€” JD Parser Refinements

- **New module**: `jd/jd_parser.py` â€” regex/heuristic JD parser with CLI entry point
- Section detection for conventional and unconventional headings
- Expanded technology vocabulary and concept keyword extraction
- Hyphen-normalised keyword deduplication and experience range parsing

### v1.0.0 â€” Initial Release

- Deterministic keyword-evidence scoring pipeline
- Additive 0â€“100 scoring model with six components
- Evidence localisation, credibility assessment, consistency engine
- Dataset and JD skill audit modules
- Structured output: diagnostics.json, run_summary.json, ranked_candidates.csv, pipeline.log

