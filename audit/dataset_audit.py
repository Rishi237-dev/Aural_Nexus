from collections import Counter
from statistics import mean
from pathlib import Path

from tqdm import tqdm


def run_audit(candidates):

    title_counter = Counter()
    skill_counter = Counter()
    industry_counter = Counter()
    country_counter = Counter()

    open_to_work_count = 0

    recruiter_response_rates = []
    github_scores = []
    notice_periods = []
    years_experience = []

    for candidate in tqdm(candidates):

        title_counter[candidate.current_title] += 1

        country_counter[
            candidate.country or "Unknown"
        ] += 1

        for skill in candidate.skills:
            skill_counter[skill["name"]] += 1

        years_experience.append(
            candidate.years_experience
        )

        signals = candidate.signals

        if signals.get("open_to_work_flag"):
            open_to_work_count += 1

        recruiter_response_rates.append(
            signals.get(
                "recruiter_response_rate",
                0
            )
        )

        github_score = signals.get(
            "github_activity_score",
            -1
        )

        if github_score != -1:
            github_scores.append(
                github_score
            )

        notice_periods.append(
            signals.get(
                "notice_period_days",
                0
            )
        )

        for job in candidate.career_history:

            industry_counter[
                job.get(
                    "industry",
                    "Unknown"
                )
            ] += 1

    report = []

    report.append("=" * 60)

    report.append("\nTOP 20 TITLES")
    for title, count in title_counter.most_common(20):
        report.append(
            f"{title}: {count}"
        )

    report.append("\nTOP 50 SKILLS")
    for skill, count in skill_counter.most_common(50):
        report.append(
            f"{skill}: {count}"
        )

    report.append("\nTOP 20 INDUSTRIES")
    for industry, count in industry_counter.most_common(20):
        report.append(
            f"{industry}: {count}"
        )

    report.append("\nTOP COUNTRIES")
    for country, count in country_counter.most_common(20):
        report.append(
            f"{country}: {count}"
        )

    report.append("\n" + "=" * 60)

    report.append(
        f"\nCandidates: {len(candidates)}"
    )

    report.append(
        f"Open To Work %: "
        f"{100 * open_to_work_count / len(candidates):.2f}"
    )

    report.append(
        f"Avg Experience: "
        f"{mean(years_experience):.2f}"
    )

    report.append(
        f"Avg Recruiter Response Rate: "
        f"{mean(recruiter_response_rates):.3f}"
    )

    if github_scores:
        report.append(
            f"Avg Github Score: "
            f"{mean(github_scores):.2f}"
        )

    report.append(
        f"Avg Notice Period: "
        f"{mean(notice_periods):.2f}"
    )

    report.append("\n" + "=" * 60)

    report_text = "\n".join(report)

    Path("outputs").mkdir(
        exist_ok=True
    )

    with open(
        "outputs/audit_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    print("\nAudit Complete")
    print(
        "Saved to outputs/audit_report.txt"
    )