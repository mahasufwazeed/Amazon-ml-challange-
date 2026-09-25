"""Command-Line Interface to run the Amazon ML Challenge 2026 Pipeline.

Usage:
    python run_pipeline.py --train --predict --validate
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "business_entity_resolution"))

from src.config import ERConfig
from src.pipeline import ERPipeline
from src.validate import validate_submission_files


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Business Entity Resolution")
    parser.add_argument("--train", action="store_true", help="Run model training on train split")
    parser.add_argument("--predict", action="store_true", help="Run inference on test split")
    parser.add_argument("--validate", action="store_true", help="Validate output files against rules")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for fast training")
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"), help="Path to train directory")
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"), help="Path to test directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Path to output directory")

    args = parser.parse_args()

    config = ERConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
    )
    pipeline = ERPipeline(config=config)

    # 1. Training Phase
    if args.train:
        s1_train = args.train_dir / "train_source1.tsv"
        s2_train = args.train_dir / "train_source2.tsv"
        s3_train = args.train_dir / "train_source3.tsv"
        gt_train = args.train_dir / "train_ground_truth.tsv"

        if not s1_train.exists():
            print(f"Error: Training file not found: {s1_train}")
            sys.exit(1)

        pipeline.train_and_validate(
            s1_path=s1_train,
            s2_path=s2_train,
            s3_path=s3_train,
            gt_path=gt_train,
            sample_size=args.sample_size,
        )

    # 2. Prediction Phase
    if args.predict:
        s1_test = args.test_dir / "test_source1.tsv"
        s2_test = args.test_dir / "test_source2.tsv"
        s3_test = args.test_dir / "test_source3.tsv"

        if not s1_test.exists():
            print(f"Error: Test file not found: {s1_test}")
            sys.exit(1)

        pipeline.run_inference(
            s1_path=s1_test,
            s2_path=s2_test,
            s3_path=s3_test,
            output_dir=args.output_dir,
        )

    # 3. Validation Phase
    if args.validate:
        matching_file = args.output_dir / "matching_results.tsv"
        candidate_file = args.output_dir / "candidate_pairs.tsv"

        print("Validating submission files...")
        errors = validate_submission_files(
            matching_file=matching_file,
            candidate_file=candidate_file,
            test_dir=args.test_dir,
        )

        if not errors:
            print("============================================================")
            print("VALIDATION SUCCESSFUL: PASS (exit 0)")
            print("The generated submission package is 100% compliant!")
            print("============================================================")
        else:
            print("============================================================")
            print(f"VALIDATION FAILED: {len(errors)} issues identified:")
            for i, err in enumerate(errors, 1):
                print(f"{i}. {err}")
            print("============================================================")
            sys.exit(1)


if __name__ == "__main__":
    main()
