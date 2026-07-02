# AI-Assisted Recruitment System

## Architecture Overview

### Project Objective

Build an AI-assisted recruitment system capable of retrieving and ranking the most relevant candidates for a given Job Description (JD).

The system must go beyond traditional ATS keyword matching by combining lexical retrieval, semantic retrieval, structured metadata, and Redrob behavioral signals.

The project is divided into two independent stages:

1. Candidate Retrieval
2. Candidate Ranking

Only the retrieval stage is being implemented now.

---

# Overall Pipeline

```text
Raw Candidate JSONL
        │
        ▼
Candidate Gate
(Removes only clearly irrelevant candidates)
        │
        ▼
Filtered Candidate JSONL
(~85,000 candidates)
        │
        ▼
Hybrid Candidate Retrieval
        │
        ▼
Top 2,000–4,000 Candidates
        │
        ▼
Hybrid Ranking Engine
(Not implemented yet)
        │
        ▼
Final Top 100
```

---

# Candidate Gate

Status

Completed

Purpose

Remove only candidates with extremely low probability of matching.

The gate intentionally favors recall over precision.

Candidates are rejected only for reasons such as

* clearly insufficient experience
* explicit negative requirements
* mandatory hard constraint failures
* consulting-only careers (when JD explicitly disallows)
* pure research with no production experience
* pure CV/Speech profiles with no relevant NLP/IR work

Technology keyword matching is NOT used for rejection.

---

# Current Dataset

Input

```text
filtered_candidates.jsonl
```

Contains approximately

```text
85,000 candidates
```

Every line is an independent JSON object.

Processing must always be performed using streaming.

Never load the entire JSONL into memory.

---

# Retrieval Stage

The retrieval stage is responsible for reducing

```text
85,000
```

to approximately

```text
2,000–4,000
```

without sacrificing recall.

Retrieval is NOT ranking.

No evidence scoring.

No consistency scoring.

No explanation generation.

No career progression scoring.

---

# Candidate Retrieval Documents

Each candidate is transformed into two retrieval documents.

## Semantic Document

Purpose

Embedding generation.

Contains

* Current title
* Professional summary
* Career history
* Work descriptions
* Education
* Technical skills

Removes

* Name
* Country
* Location
* Company size
* Salary
* Recruiter popularity
* Views
* Connections
* Other irrelevant metadata

---

## Lexical Document

Purpose

BM25 retrieval.

Contains

* Technologies
* Frameworks
* Libraries
* Databases
* Cloud
* ML concepts
* Programming languages

The lexical document is optimized for exact token matching.

---

# Embedding Model

Model

```text
BAAI/bge-base-en-v1.5
```

Purpose

Dense semantic retrieval.

Candidate embeddings are generated offline.

Embeddings are persisted.

The same embedding is never recomputed.

---

# Vector Database

FAISS

Recommended index

```text
IndexFlatIP
```

Embeddings must be normalized before indexing.

---

# Lexical Retrieval

Library

```text
rank_bm25
```

Input

Lexical retrieval document.

---

# Hybrid Retrieval

Two independent retrieval systems run.

1.

Dense Retrieval

using

* BGE embeddings
* FAISS

2.

Sparse Retrieval

using

* BM25

Each produces an independent ranked list.

---

# Fusion

The ranked lists are merged using

```text
Reciprocal Rank Fusion (RRF)
```

Manual weighting should be avoided.

---

# Redrob Signals

Redrob signals are NOT retrieval signals.

They are confidence modifiers.

Only objective availability signals should be used.

Allowed

* profile_completeness_score
* open_to_work_flag
* last_active_date
* interview_completion_rate
* verified_email
* verified_phone

Do NOT use

* recruiter popularity
* profile views
* recruiter saves
* connection count

These are not measures of candidate relevance.

Redrob adjusts retrieval confidence only.

It should never dominate semantic similarity.

---

# Retrieval Output

The retrieval stage returns approximately

```text
2,000–4,000 candidates
```

Each candidate should contain

* candidate_id
* dense retrieval score
* BM25 score
* Reciprocal Rank Fusion score
* Redrob adjustment
* final retrieval score

sorted descending.

---

# Ranking Stage

Not implemented yet.

Ranking will later combine

* Evidence Locator
* Consistency Engine
* Career Progression
* Redrob signals
* Explainability
* Additional ranking features

Ranking is intentionally separated from retrieval.

---

# Design Principles

* CPU compatible
* Streaming JSONL processing
* Modular architecture
* Offline embedding generation
* No runtime LLM
* No LangChain
* No LlamaIndex
* Reproducible results
* High recall during retrieval
* High precision during ranking


