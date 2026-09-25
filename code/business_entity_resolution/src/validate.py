"""Submission validation tool for Amazon ML Challenge 2026.

Uses standard Python library only (no external dependencies).
Ensures 100% adherence to all submission rules before upload:
1. File existence and TSV tab delimiter
2. Header format and columns
3. Exact 1-to-1 match with test_source1.tsv entity IDs (including France)
4. No self-matches (S1- prefix prohibited in matched lists)
5. Only valid S2- and S3- IDs that exist in test set
6. No duplicate entity IDs within any row
7. Strict subset rule: all matches in matching_results.tsv must appear in candidate_pairs.tsv
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_tsv_line(line: str) -> List[str]:
    """Split line by tab without quotes."""
    return line.rstrip("\r\n").split("\t")


def load_test_ids(test_dir: Path) -> Tuple[List[str], Set[str], Set[str]]:
    """Load valid IDs from test_source1.tsv, test_source2.tsv, test_source3.tsv."""
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    if not s1_path.exists():
        raise FileNotFoundError(f"Missing {s1_path}")
    if not s2_path.exists():
        raise FileNotFoundError(f"Missing {s2_path}")
    if not s3_path.exists():
        raise FileNotFoundError(f"Missing {s3_path}")

    # Read S1 IDs in order
    s1_ids = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = parse_tsv_line(f.readline())
        try:
            id_idx = header.index("entity_id")
        except ValueError:
            id_idx = 0
        for line in f:
            if not line.strip():
                continue
            parts = parse_tsv_line(line)
            if len(parts) > id_idx:
                s1_ids.append(parts[id_idx].strip())

    # Read S2 IDs
    s2_ids = set()
    with open(s2_path, "r", encoding="utf-8") as f:
        header = parse_tsv_line(f.readline())
        try:
            id_idx = header.index("entity_id")
        except ValueError:
            id_idx = 0
        for line in f:
            if not line.strip():
                continue
            parts = parse_tsv_line(line)
            if len(parts) > id_idx:
                s2_ids.add(parts[id_idx].strip())

    # Read S3 IDs
    s3_ids = set()
    with open(s3_path, "r", encoding="utf-8") as f:
        header = parse_tsv_line(f.readline())
        try:
            id_idx = header.index("entity_id")
        except ValueError:
            id_idx = 0
        for line in f:
            if not line.strip():
                continue
            parts = parse_tsv_line(line)
            if len(parts) > id_idx:
                s3_ids.add(parts[id_idx].strip())

    return s1_ids, s2_ids, s3_ids


def validate_submission_files(
    matching_file: Path,
    candidate_file: Path,
    test_dir: Path,
) -> List[str]:
    """Perform exhaustive validation checks against competition rules."""
    errors = []

    if not matching_file.exists():
        return [f"Matching file not found: {matching_file}"]
    if not candidate_file.exists():
        return [f"Candidate file not found: {candidate_file}"]

    # 1. Load reference test IDs
    try:
        s1_expected_ids, s2_valid_ids, s3_valid_ids = load_test_ids(test_dir)
        valid_target_ids = s2_valid_ids | s3_valid_ids
    except Exception as e:
        return [f"Failed to read test directory: {e}"]

    expected_s1_count = len(s1_expected_ids)
    expected_s1_set = set(s1_expected_ids)

    # 2. Validate candidate_pairs.tsv
    candidate_dict: Dict[str, Set[str]] = {}
    candidate_order = []
    with open(candidate_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        cand_header = parse_tsv_line(first_line)
        if len(cand_header) != 2 or cand_header[0] != "source1_entity_id" or cand_header[1] != "candidate_entity_ids":
            errors.append(
                f"Invalid candidate_pairs.tsv header: {cand_header}. Expected ['source1_entity_id', 'candidate_entity_ids']"
            )

        line_num = 1
        for line in f:
            line_num += 1
            if not line.strip():
                continue
            parts = parse_tsv_line(line)
            if len(parts) == 1:
                parts.append("")
            if len(parts) != 2:
                errors.append(f"candidate_pairs.tsv line {line_num}: Expected 2 tab-separated columns, got {len(parts)}")
                continue

            s1_id, cand_str = parts[0].strip(), parts[1].strip()
            if s1_id in candidate_dict:
                errors.append(f"candidate_pairs.tsv line {line_num}: Duplicate source1_entity_id '{s1_id}'")
            candidate_order.append(s1_id)

            cand_list = [c.strip() for c in cand_str.split(",") if c.strip()]
            if len(cand_list) != len(set(cand_list)):
                errors.append(f"candidate_pairs.tsv line {line_num}: Duplicate candidate IDs in list for '{s1_id}'")

            for cid in cand_list:
                if cid.startswith("S1-"):
                    errors.append(f"candidate_pairs.tsv line {line_num}: Self-match '{cid}' in candidates for '{s1_id}'")
                elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                    errors.append(f"candidate_pairs.tsv line {line_num}: Invalid prefix for candidate '{cid}'")
                elif cid not in valid_target_ids:
                    errors.append(f"candidate_pairs.tsv line {line_num}: Candidate '{cid}' does not exist in test set")

            candidate_dict[s1_id] = set(cand_list)

    if len(candidate_order) != expected_s1_count:
        errors.append(
            f"candidate_pairs.tsv row count mismatch: found {len(candidate_order)}, expected {expected_s1_count}"
        )

    # 3. Validate matching_results.tsv
    matching_dict: Dict[str, Set[str]] = {}
    matching_order = []
    with open(matching_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        match_header = parse_tsv_line(first_line)
        if len(match_header) != 2 or match_header[0] != "source1_entity_id" or match_header[1] != "matched_entity_ids":
            errors.append(
                f"Invalid matching_results.tsv header: {match_header}. Expected ['source1_entity_id', 'matched_entity_ids']"
            )

        line_num = 1
        for line in f:
            line_num += 1
            if not line.strip():
                continue
            parts = parse_tsv_line(line)
            if len(parts) == 1:
                parts.append("")
            if len(parts) != 2:
                errors.append(f"matching_results.tsv line {line_num}: Expected 2 tab-separated columns, got {len(parts)}")
                continue

            s1_id, match_str = parts[0].strip(), parts[1].strip()
            if s1_id in matching_dict:
                errors.append(f"matching_results.tsv line {line_num}: Duplicate source1_entity_id '{s1_id}'")
            matching_order.append(s1_id)

            match_list = [m.strip() for m in match_str.split(",") if m.strip()]
            if len(match_list) != len(set(match_list)):
                errors.append(f"matching_results.tsv line {line_num}: Duplicate matched IDs in list for '{s1_id}'")

            for mid in match_list:
                if mid.startswith("S1-"):
                    errors.append(f"matching_results.tsv line {line_num}: Self-match '{mid}' in matches for '{s1_id}'")
                elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                    errors.append(f"matching_results.tsv line {line_num}: Invalid prefix for matched ID '{mid}'")
                elif mid not in valid_target_ids:
                    errors.append(f"matching_results.tsv line {line_num}: Matched ID '{mid}' does not exist in test set")

            matching_dict[s1_id] = set(match_list)

    if len(matching_order) != expected_s1_count:
        errors.append(
            f"matching_results.tsv row count mismatch: found {len(matching_order)}, expected {expected_s1_count}"
        )

    # 4. Check that all test S1 IDs are present
    missing_in_match = expected_s1_set - set(matching_dict.keys())
    if missing_in_match:
        sample = list(missing_in_match)[:5]
        errors.append(f"matching_results.tsv is missing {len(missing_in_match)} test S1 entities (sample: {sample})")

    # 5. Check subset condition: every matched ID must be in candidates
    for s1_id, matches in matching_dict.items():
        candidates = candidate_dict.get(s1_id, set())
        invalid_matches = matches - candidates
        if invalid_matches:
            errors.append(
                f"Entity '{s1_id}' has matches not present in candidate_pairs.tsv: {list(invalid_matches)[:5]}"
            )
            if len(errors) > 25:
                errors.append("... too many errors, stopping check.")
                break

    return errors


def main():
    parser = argparse.ArgumentParser(description="Submission Validator for Amazon ML Challenge 2026")
    parser.add_argument("--matching", required=True, type=Path, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, type=Path, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, type=Path, help="Path to dataset/test directory")

    args = parser.parse_args()

    errors = validate_submission_files(args.matching, args.candidate, args.test_dir)

    if not errors:
        print("=" * 60)
        print("VALIDATION SUCCESSFUL: PASS (exit 0)")
        print("The files strictly satisfy all format and integrity constraints.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print(f"VALIDATION FAILED: {len(errors)} issues identified (exit 1):")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
