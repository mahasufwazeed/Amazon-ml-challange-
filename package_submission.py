"""Automated packaging script for Amazon ML Challenge 2026.

Creates <team_name>_submission.zip adhering strictly to competition specifications:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import argparse
import sys
import zipfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "business_entity_resolution"))
from src.validate import validate_submission_files


def create_submission_zip(team_name: str, base_dir: Path, test_dir: Path) -> Path:
    output_dir = base_dir / "output"
    code_dir = base_dir / "code" / "business_entity_resolution"
    doc_path = base_dir / "Documentation_template.md"

    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    print("Step 1: Validating submission files against official competition rules...")
    errors = validate_submission_files(
        matching_file=matching_file,
        candidate_file=candidate_file,
        test_dir=test_dir,
    )
    if errors:
        print("ERROR: Validation failed! Cannot create package until issues are fixed:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)
    print("PASS: Both matching_results.tsv and candidate_pairs.tsv passed all validation rules.\n")

    # Clean zip file name
    clean_team_name = team_name.strip().replace(" ", "_").lower()
    zip_path = base_dir / f"{clean_team_name}_submission.zip"

    print(f"Step 2: Creating archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(matching_file, arcname="output/matching_results.tsv")
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")
        print("  + Added output/matching_results.tsv")
        print("  + Added output/candidate_pairs.tsv")

        # 2. Documentation template
        if doc_path.exists():
            zf.write(doc_path, arcname="Documentation_template.md")
            print("  + Added Documentation_template.md")
        else:
            print("  ! Warning: Documentation_template.md not found.")

        # 3. Code folder
        for file_path in code_dir.rglob("*"):
            if file_path.is_file():
                # Skip cache and temporary files
                if "__pycache__" in file_path.parts or ".pytest_cache" in file_path.parts or file_path.suffix == ".pyc":
                    continue
                rel_path = file_path.relative_to(base_dir)
                zf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                print(f"  + Added {rel_path}")

    print(f"\nSUCCESS: Submission archive created at: {zip_path}")
    print(f"Archive size: {zip_path.stat().st_size / 1024:.1f} KB")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge 2026 Submission")
    parser.add_argument("--team-name", default="antigravity_team", help="Your registered team name")
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"), help="Path to test set directory")

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    create_submission_zip(team_name=args.team_name, base_dir=base_dir, test_dir=args.test_dir)


if __name__ == "__main__":
    main()
