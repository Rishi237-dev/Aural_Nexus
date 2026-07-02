import json

from candidate.schema import Candidate
from candidate.candidate_io import (
    write_schema_json,
    write_parsed_candidates_json,
)


def parse_candidate(raw: dict) -> Candidate:

    profile = raw["profile"]

    return Candidate(
        candidate_id=raw["candidate_id"],

        anonymized_name=profile.get(
            "anonymized_name", ""
        ),

        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),

        location=profile.get("location", ""),

        country=profile.get("country", ""),

        years_experience=profile.get(
            "years_of_experience", 0
        ),

        current_title=profile.get(
            "current_title", ""
        ),

        current_company=profile.get(
            "current_company", ""
        ),

        current_company_size=profile.get(
            "current_company_size", ""
        ),

        current_industry=profile.get(
            "current_industry", ""
        ),

        # Full skill dicts — name, proficiency, endorsements, duration_months
        skills=raw.get("skills", []),

        career_history=raw.get(
            "career_history", []
        ),

        education=raw.get(
            "education", []
        ),

        certifications=raw.get(
            "certifications", []
        ),

        languages=raw.get(
            "languages", []
        ),

        signals=raw.get(
            "redrob_signals", {}
        )
    )


def load_json_candidates(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return [
        parse_candidate(c)
        for c in data
    ]


def load_jsonl_candidates(path):

    candidates = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            raw = json.loads(line)

            candidates.append(
                parse_candidate(raw)
            )

    return candidates


def load_candidates(path):

    if path.endswith(".jsonl"):
        candidates = load_jsonl_candidates(path)
    else:
        candidates = load_json_candidates(path)

    write_schema_json()
    write_parsed_candidates_json(candidates)

    return candidates


def build_candidate_text(candidate):

    # Extract skill names from full skill dicts
    skills_text = " ".join(
        s["name"] for s in candidate.skills
    )

    career_text = " ".join(
        job["description"]
        for job in candidate.career_history
    )

    return f"""
    {candidate.headline}

    {candidate.summary}

    {skills_text}

    {career_text}
    """