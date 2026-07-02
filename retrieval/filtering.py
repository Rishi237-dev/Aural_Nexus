import argparse
import json
from pathlib import Path


def filter_candidates(
    gate_output_path: str,
    input_jsonl_path: str,
    output_jsonl_path: str,
) -> None:
    """
    Creates a new JSONL containing only candidates
    that passed the Candidate Gate.
    """

    print("Loading gate results...")

    with open(gate_output_path, "r", encoding="utf-8") as f:
        gate_results = json.load(f)

    keep_ids = {
        candidate["candidate_id"]
        for candidate in gate_results
        if candidate.get("passed_filter", False)
    }

    print(f"Candidates to keep : {len(keep_ids):,}")

    kept = 0
    processed = 0

    with open(input_jsonl_path, "r", encoding="utf-8") as infile, \
         open(output_jsonl_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            processed += 1
            candidate = json.loads(line)
            if candidate["candidate_id"] in keep_ids:
                outfile.write(json.dumps(candidate))
                outfile.write("\n")
                kept += 1

    print("\nDone.")
    print(f"Processed : {processed:,}")
    print(f"Kept      : {kept:,}")
    print(f"Saved to  : {output_jsonl_path}")


if __name__ == "__main__":
    resume_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Filter candidate JSONL by gate output.")
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=resume_root / "output.json",
        help="Path to the gate output JSON file.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=resume_root / "provided_info" / "candidates.jsonl",
        help="Path to the full candidates JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=resume_root / "retrieval" / "filtered_candidates.jsonl",
        help="Destination path for the filtered JSONL file.",
    )

    args = parser.parse_args()
    filter_candidates(str(args.gate_output), str(args.input_jsonl), str(args.output_jsonl))