from dataclasses import dataclass


@dataclass
class Candidate:

    candidate_id: str

    anonymized_name: str

    headline: str
    summary: str

    location: str
    country: str

    years_experience: float

    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    skills: list           # list of full skill dicts (name, proficiency, endorsements, duration_months)

    career_history: list

    education: list

    certifications: list

    languages: list

    signals: dict