"""End-to-end Entity Resolution Pipeline for Amazon ML Challenge 2026.

Integrates blocking, feature extraction, model inference, and output validation.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ERConfig, default_config
from .preprocessing import clean_business_name, clean_address, normalize_country
from .blocking import MultiIndexBlocker, get_char_ngrams
from .features import extract_pair_features
from .metrics import evaluate_macro_f_beta
from .model import ERClassifier
from .validate import validate_submission_files


class ERPipeline:
    """Master pipeline managing end-to-end training, inference, and packaging."""

    def __init__(self, config: Optional[ERConfig] = None):
        self.config = config or default_config
        self.config.resolve_paths()
        self.blocker = MultiIndexBlocker(
            max_candidates_per_s1=self.config.max_candidates_per_s1,
            min_ngram_overlap=self.config.ngram_overlap_threshold,
        )
        self.model = ERClassifier(config=self.config)

    def load_ground_truth(self, gt_path: Path) -> Dict[str, List[str]]:
        """Load ground truth mapping from TSV file."""
        df_gt = pd.read_csv(gt_path, sep="\t")
        df_gt["matched_entity_ids"] = df_gt["matched_entity_ids"].fillna("")

        gt_dict = {}
        for row in df_gt.itertuples():
            s1_id = row.source1_entity_id
            m_str = getattr(row, "matched_entity_ids", "")
            gt_dict[s1_id] = [m.strip() for m in m_str.split(",") if m.strip()]

        return gt_dict

    def train_and_validate(
        self,
        s1_path: Path,
        s2_path: Path,
        s3_path: Path,
        gt_path: Path,
        sample_size: Optional[int] = None,
    ):
        """Run candidate generation, featurization, and model training with CV validation."""
        print(f"Loading training datasets from {s1_path.parent}...")
        df_s1 = pd.read_csv(s1_path, sep="\t")
        df_s2 = pd.read_csv(s2_path, sep="\t")
        df_s3 = pd.read_csv(s3_path, sep="\t")
        gt_dict = self.load_ground_truth(gt_path)

        if sample_size and sample_size < len(df_s1):
            print(f"Subsampling to {sample_size:,} S1 entities for rapid development/CV...")
            df_s1 = df_s1.sample(n=sample_size, random_state=self.config.random_seed)
            gt_dict = {eid: gt_dict.get(eid, []) for eid in df_s1["entity_id"]}

        # 1. Index S2 and S3
        self.blocker.index_target_records(df_s2, source_label="Source 2")
        self.blocker.index_target_records(df_s3, source_label="Source 3")

        # 2. Block S1
        print("Running blocking on S1 entities...")
        candidates_by_s1 = self.blocker.block_s1_dataframe(df_s1)

        # 3. Create feature pairs for training
        print("Extracting feature representations for training pairs...")
        rows = []
        labels = []

        # Pre-process S1 records
        s1_records = {}
        for row in df_s1.itertuples():
            clean_name, core_name, sorted_tokens = clean_business_name(row.business_name)
            clean_addr, postal_code, numbers = clean_address(getattr(row, "business_address", ""))
            ngrams = get_char_ngrams(core_name, n=3)
            s1_records[row.entity_id] = {
                "clean_name": clean_name,
                "core_name": core_name,
                "sorted_tokens": sorted_tokens,
                "clean_addr": clean_addr,
                "postal_code": postal_code,
                "numbers": numbers,
                "ngrams": ngrams,
            }

        candidate_rows_for_tuning = []

        for s1_id, cand_list in tqdm(candidates_by_s1.items(), desc="Featurizing training pairs"):
            s1_data = s1_records[s1_id]
            true_set = set(gt_dict.get(s1_id, []))
            total_cand = len(cand_list)

            for rank, cid in enumerate(cand_list):
                if cid not in self.blocker.target_records:
                    continue
                target_data = dict(self.blocker.target_records[cid])
                target_data["entity_id"] = cid

                feat = extract_pair_features(
                    s1_data=s1_data,
                    target_data=target_data,
                    candidate_rank=rank,
                    total_candidates=total_cand,
                )
                feat["source1_entity_id"] = s1_id
                feat["candidate_entity_id"] = cid
                is_match = 1 if cid in true_set else 0
                rows.append(feat)
                labels.append(is_match)
                candidate_rows_for_tuning.append(feat)

            # Ensure negative samples exist for robust binary classification
            non_matches = [cid for cid in cand_list if cid not in true_set]
            if len(non_matches) < 3:
                country = df_s1.loc[df_s1["entity_id"] == s1_id, "country"].values[0]
                country_norm = normalize_country(country)
                all_country_targets = [
                    tid for tid, rec in self.blocker.target_records.items()
                    if rec["country"] == country_norm and tid not in true_set and tid not in cand_list
                ]
                if all_country_targets:
                    sampled_negs = random.sample(all_country_targets, min(len(all_country_targets), 3))
                    for offset, nid in enumerate(sampled_negs):
                        target_data = dict(self.blocker.target_records[nid])
                        target_data["entity_id"] = nid
                        feat = extract_pair_features(
                            s1_data=s1_data,
                            target_data=target_data,
                            candidate_rank=total_cand + offset,
                            total_candidates=total_cand + len(sampled_negs),
                        )
                        feat["source1_entity_id"] = s1_id
                        feat["candidate_entity_id"] = nid
                        rows.append(feat)
                        labels.append(0)
                        candidate_rows_for_tuning.append(feat)

        df_train_pairs = pd.DataFrame(rows)
        y_train = np.array(labels)
        feature_cols = [
            c for c in df_train_pairs.columns if c not in ["source1_entity_id", "candidate_entity_id"]
        ]

        print(
            f"Training dataset: {len(df_train_pairs):,} pairs, "
            f"Positives={y_train.sum():,} ({y_train.mean()*100:.2f}%)"
        )

        # 4. Train LightGBM model
        self.model.train(df_train_pairs[feature_cols], y_train)

        # 5. Calibrate thresholds for Macro F_0.5
        df_tuning = pd.DataFrame(candidate_rows_for_tuning)
        self.model.calibrate_thresholds(
            df_candidates=df_tuning,
            ground_truth=gt_dict,
            feature_cols=feature_cols,
        )

        # Save model
        model_save_path = self.config.model_dir / "er_lgbm_model.joblib"
        self.model.save(model_save_path)
        return self.model

    def run_inference(
        self,
        s1_path: Path,
        s2_path: Path,
        s3_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Run candidate generation, model scoring, and generate submission files."""
        out_dir = output_dir or self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        cand_out_path = out_dir / self.config.candidate_output_filename
        match_out_path = out_dir / self.config.matching_output_filename

        print(f"Loading test datasets from {s1_path.parent}...")
        df_test_s1 = pd.read_csv(s1_path, sep="\t")
        df_test_s2 = pd.read_csv(s2_path, sep="\t")
        df_test_s3 = pd.read_csv(s3_path, sep="\t")

        print(f"Test counts: S1={len(df_test_s1):,}, S2={len(df_test_s2):,}, S3={len(df_test_s3):,}")

        # Index test S2 and S3 (including France and all open countries)
        test_blocker = MultiIndexBlocker(
            max_candidates_per_s1=self.config.max_candidates_per_s1,
            min_ngram_overlap=self.config.ngram_overlap_threshold,
        )
        test_blocker.index_target_records(df_test_s2, source_label="Test Source 2")
        test_blocker.index_target_records(df_test_s3, source_label="Test Source 3")

        # 1. Generate Candidates
        print("Generating candidate pairs for test Source 1 entities...")
        candidates_by_s1 = test_blocker.block_s1_dataframe(df_test_s1)

        # Save candidate_pairs.tsv
        print(f"Writing {cand_out_path}...")
        with open(cand_out_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for row in df_test_s1.itertuples():
                eid = row.entity_id
                cands = candidates_by_s1.get(eid, [])
                f.write(f"{eid}\t{','.join(cands)}\n")

        # 2. Extract Features for candidates
        print("Extracting features for candidate pairs...")
        rows = []
        s1_records = {}
        for row in df_test_s1.itertuples():
            clean_name, core_name, sorted_tokens = clean_business_name(row.business_name)
            clean_addr, postal_code, numbers = clean_address(getattr(row, "business_address", ""))
            ngrams = get_char_ngrams(core_name, n=3)
            s1_records[row.entity_id] = {
                "clean_name": clean_name,
                "core_name": core_name,
                "sorted_tokens": sorted_tokens,
                "clean_addr": clean_addr,
                "postal_code": postal_code,
                "numbers": numbers,
                "ngrams": ngrams,
            }

        for s1_id, cand_list in tqdm(candidates_by_s1.items(), desc="Featurizing test candidates"):
            s1_data = s1_records[s1_id]
            total_cand = len(cand_list)
            for rank, cid in enumerate(cand_list):
                if cid not in test_blocker.target_records:
                    continue
                target_data = dict(test_blocker.target_records[cid])
                target_data["entity_id"] = cid

                feat = extract_pair_features(
                    s1_data=s1_data,
                    target_data=target_data,
                    candidate_rank=rank,
                    total_candidates=total_cand,
                )
                feat["source1_entity_id"] = s1_id
                feat["candidate_entity_id"] = cid
                rows.append(feat)

        if rows:
            df_test_pairs = pd.DataFrame(rows)
            # 3. Model scoring
            print("Predicting matches with calibrated thresholds...")
            matches_by_s1 = self.model.predict_matches(
                candidate_dict=candidates_by_s1,
                feature_matrix=df_test_pairs,
            )
        else:
            matches_by_s1 = {eid: [] for eid in df_test_s1["entity_id"]}

        # 4. Save matching_results.tsv
        print(f"Writing {match_out_path}...")
        with open(match_out_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for row in df_test_s1.itertuples():
                eid = row.entity_id
                matched_ids = matches_by_s1.get(eid, [])
                f.write(f"{eid}\t{','.join(matched_ids)}\n")

        print("Inference completed successfully.")
        return match_out_path, cand_out_path
