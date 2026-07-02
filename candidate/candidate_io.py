# ==========================================================
# candidate/candidate_io.py
#
# Responsibility: JSON serialisation of the Candidate schema
# and parsed candidate list.
#
# This module has NO parsing logic and does NOT modify the
# Candidate dataclass. It only reads already-constructed
# Candidate objects and writes them to JSON files inside
# the candidate/ directory.
#
# Output files (both overwritten on every call):
#   candidate/candidate_schema.json   — field names and types
#   candidate/parsed_candidates.json  — full list of Candidate dicts
#
# Public API
# ----------
# write_schema_json()
#     Write the Candidate dataclass field structure to
#     candidate/candidate_schema.json.
#
# write_parsed_candidates_json(candidates)
#     Serialise every Candidate in the list to
#     candidate/parsed_candidates.json.
# ==========================================================

import dataclasses
import json
from pathlib import Path

from candidate.schema import Candidate


# ------------------------------------------------------------------
# Output paths
# ------------------------------------------------------------------

_CANDIDATE_DIR     = Path("candidate")
_SCHEMA_FILE       = _CANDIDATE_DIR / "candidate_schema.json"
_CANDIDATES_FILE   = _CANDIDATE_DIR / "parsed_candidates.json"


# ------------------------------------------------------------------
# Schema serialisation
# ------------------------------------------------------------------

def write_schema_json() -> None:
    """
    Write the Candidate dataclass field structure to
    candidate/candidate_schema.json.

    The output contains the field name and its declared Python type
    for every field in the Candidate dataclass.  The schema file is
    overwritten on every call.

    Example output:
        {
          "dataclass": "Candidate",
          "fields": [
            { "name": "candidate_id", "type": "str" },
            ...
          ]
        }
    """
    fields = [
        {
            "name": f.name,
            "type": f.type.__name__ if isinstance(f.type, type) else str(f.type),
        }
        for f in dataclasses.fields(Candidate)
    ]

    payload = {
        "dataclass": "Candidate",
        "fields":    fields,
    }

    _CANDIDATE_DIR.mkdir(exist_ok=True)

    with open(_SCHEMA_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Candidate list serialisation
# ------------------------------------------------------------------

def write_parsed_candidates_json(candidates: list) -> None:
    """
    Serialise every Candidate object in *candidates* to
    candidate/parsed_candidates.json.

    Every field defined in the Candidate dataclass is included.
    The file is overwritten on every call.

    Parameters
    ----------
    candidates : list[Candidate]
        The list returned by load_candidates().
    """
    _CANDIDATE_DIR.mkdir(exist_ok=True)

    with open(_CANDIDATES_FILE, "w", encoding="utf-8") as fh:
        json.dump(
            [dataclasses.asdict(c) for c in candidates],
            fh,
            indent=2,
            ensure_ascii=False,
        )
