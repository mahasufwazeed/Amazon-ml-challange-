"""End-to-end Entity Resolution Pipeline for Amazon ML Challenge 2026.

Integrates memory-optimized country-partitioned blocking, streaming feature extraction,
LightGBM inference, and automated output validation.
"""
import csv
import gc
import os
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ERConfig, default_config
from .preprocessing import (
    clean_business_name,
    clean_address,
    normalize_country,
    extract_postal_code,
    clean_text_basic,
    extract_numbers,
)
from .blocking import CountryBlocker, MultiIndexBlocker, get_char_ngrams
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

    def load_ground_truth(
        self, gt_path: Path, filter_ids: Optional[Set[str]] = None
    ) -> Dict[str, List[str]]:
        """Load ground truth mapping from TSV file with optional ID filtering."""
        gt_dict = {}
        with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                if filter_ids is not None and s1_id not in filter_ids:
                    continue
                m_str = row[1].strip() if len(row) > 1 else ""
                gt_dict[s1_id] = [m.strip() for m in m_str.split(",") if m.strip()]
        return gt_dict

    def train_and_validate(
        self,
        s1_path: Path,
        s2_path: Path,
        s3_path: Path,
        gt_path: Path,
        sample_size: Optional[int] = 40000,
    ):
        """Run low-memory candidate generation, featurization, and model training with CV validation."""
        print(f"Loading and sampling training entities from {s1_path.parent}...")
        sample_target = sample_size if sample_size is not None else 40000
        per_country_target = max(5000, sample_target // 2)

        # 1. Balanced sampling of S1 entities across available training countries (US & India)
        s1_by_country: Dict[str, List[Tuple[str, str, str, str]]] = {"US": [], "INDIA": []}
        with open(s1_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2
            c_idx = header.index("country") if "country" in header else 3

            for row in reader:
                if len(row) <= c_idx:
                    continue
                c_norm = normalize_country(row[c_idx])
                if c_norm in s1_by_country and len(s1_by_country[c_norm]) < per_country_target:
                    s1_by_country[c_norm].append((
                        row[id_idx],
                        row[name_idx] if len(row) > name_idx else "",
                        row[addr_idx] if len(row) > addr_idx else "",
                        c_norm,
                    ))
                if all(len(v) >= per_country_target for v in s1_by_country.values()):
                    break

        sampled_s1_ids = {row[0] for rows in s1_by_country.values() for row in rows}
        print(f"Sampled {len(sampled_s1_ids):,} S1 entities ({', '.join(f'{k}: {len(v)}' for k, v in s1_by_country.items())})")

        # 2. Load ground truth for sampled entities
        gt_dict = self.load_ground_truth(gt_path, filter_ids=sampled_s1_ids)
        all_true_target_ids = {m for matches in gt_dict.values() for m in matches}
        print(f"Found {len(all_true_target_ids):,} ground truth target matches for sampled S1 entities.")

        # 3. Country-partitioned target indexing & candidate pair generation
        training_rows = []
        training_labels = []
        candidate_rows_for_tuning = []

        for country, s1_list in s1_by_country.items():
            print(f"--- Processing Training Country: {country} ---")
            blocker = CountryBlocker(
                country=country,
                max_candidates_per_s1=self.config.max_candidates_per_s1,
                min_ngram_overlap=self.config.ngram_overlap_threshold,
            )

            # Index target records (all true positive targets + up to 150,000 background records for hard negatives)
            for target_tsv in [s2_path, s3_path]:
                with open(target_tsv, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = next(reader, None)
                    id_idx = header.index("entity_id") if "entity_id" in header else 0
                    name_idx = header.index("business_name") if "business_name" in header else 1
                    addr_idx = header.index("business_address") if "business_address" in header else 2
                    c_idx = header.index("country") if "country" in header else 3

                    bg_count = 0
                    for row in reader:
                        if len(row) <= c_idx:
                            continue
                        if normalize_country(row[c_idx]) == country:
                            eid = row[id_idx]
                            is_true = eid in all_true_target_ids
                            if is_true or bg_count < 150000:
                                blocker.add_target(
                                    eid=eid,
                                    name=row[name_idx] if len(row) > name_idx else "",
                                    address=row[addr_idx] if len(row) > addr_idx else "",
                                )
                                if not is_true:
                                    bg_count += 1

            print(f"Indexed {len(blocker):,} target records for {country}.")

            # Block S1 records and generate candidate pairs
            for s1_id, name, addr, c_norm in tqdm(s1_list, desc=f"Featurizing {country} pairs"):
                clean_name, core_name, sorted_tokens = clean_business_name(name)
                postal_code = extract_postal_code(addr)
                clean_addr = clean_text_basic(addr)
                ngrams = get_char_ngrams(core_name, n=3)
                s1_data = {
                    "clean_name": clean_name,
                    "core_name": core_name,
                    "sorted_tokens": sorted_tokens,
                    "clean_addr": clean_addr,
                    "postal_code": postal_code,
                    "numbers": extract_numbers(clean_addr),
                    "ngrams": ngrams,
                }
                cand_indices = blocker.generate_candidate_indices(
                    clean_name=clean_name,
                    core_name=core_name,
                    sorted_tokens=sorted_tokens,
                    postal_code=postal_code,
                    ngrams=ngrams,
                )
                true_set = set(gt_dict.get(s1_id, []))
                total_cands = len(cand_indices)

                for rank, idx in enumerate(cand_indices):
                    t_data = blocker.get_target_dict_by_idx(idx)
                    feat = extract_pair_features(
                        s1_data=s1_data,
                        target_data=t_data,
                        candidate_rank=rank,
                        total_candidates=total_cands,
                    )
                    feat["source1_entity_id"] = s1_id
                    feat["candidate_entity_id"] = t_data["entity_id"]
                    is_match = 1 if t_data["entity_id"] in true_set else 0
                    training_rows.append(feat)
                    training_labels.append(is_match)
                    candidate_rows_for_tuning.append(feat)

            # Free memory for this country
            blocker.clear()
            del blocker
            gc.collect()

        df_train_pairs = pd.DataFrame(training_rows)
        y_train = np.array(training_labels)
        feature_cols = [
            c for c in df_train_pairs.columns if c not in ["source1_entity_id", "candidate_entity_id"]
        ]

        print(
            f"Training dataset ready: {len(df_train_pairs):,} pairs, "
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

        # 6. Save model
        model_save_path = self.config.model_dir / "er_lgbm_model.joblib"
        self.model.save(model_save_path)
        return self.model

    def run_inference(
        self,
        s1_path: Path,
        s2_path: Path,
        s3_path: Path,
        output_dir: Optional[Path] = None,
        batch_size: int = 10000,
    ) -> Tuple[Path, Path]:
        """Run candidate generation, model scoring, and generate submission files using country partitioning."""
        out_dir = output_dir or self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        cand_out_path = out_dir / self.config.candidate_output_filename
        match_out_path = out_dir / self.config.matching_output_filename

        # Ensure model is loaded
        model_save_path = self.config.model_dir / "er_lgbm_model.joblib"
        if self.model.model is None:
            if model_save_path.exists():
                print(f"Loading trained matcher from {model_save_path}...")
                self.model.load(model_save_path)
            else:
                print("No pre-trained model found. Auto-training matcher on representative split...")
                train_dir = self.config.train_dir
                self.train_and_validate(
                    s1_path=train_dir / "train_source1.tsv",
                    s2_path=train_dir / "train_source2.tsv",
                    s3_path=train_dir / "train_source3.tsv",
                    gt_path=train_dir / "train_ground_truth.tsv",
                )

        print(f"Scanning test countries from {s1_path}...")
        country_counts = {}
        with open(s1_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            c_idx = header.index("country") if "country" in header else 3
            for row in reader:
                if len(row) > c_idx:
                    c = normalize_country(row[c_idx])
                    country_counts[c] = country_counts.get(c, 0) + 1

        # Process smallest country first for rapid feedback (e.g. France -> US -> India)
        sorted_countries = sorted(country_counts.keys(), key=lambda c: country_counts[c])
        print(f"Discovered test countries: {sorted_countries} (Entity counts: {country_counts})")

        # Set up temporary SQLite database to buffer results without high memory
        temp_db_path = out_dir / "temp_inference_buffer.db"
        if temp_db_path.exists():
            temp_db_path.unlink()

        conn = sqlite3.connect(temp_db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS results (source1_entity_id TEXT PRIMARY KEY, candidates TEXT, matches TEXT)"
        )

        # Process each country partition independently
        feature_cols = self.model.feature_names
        total_s1_processed = 0

        for country in sorted_countries:
            total_country_s1 = country_counts[country]
            print(f"\n============================================================")
            print(f"Processing Country Partition: {country} ({total_country_s1:,} S1 entities)")
            print(f"============================================================")

            # Build country blocker
            country_blocker = CountryBlocker(
                country=country,
                max_candidates_per_s1=self.config.max_candidates_per_s1,
                min_ngram_overlap=self.config.ngram_overlap_threshold,
            )

            print(f"Indexing {country} records from Test Source 2...")
            country_blocker.index_from_tsv(s2_path)
            print(f"Indexing {country} records from Test Source 3...")
            country_blocker.index_from_tsv(s3_path)
            print(f"Total target records indexed for {country}: {len(country_blocker):,}")

            # Stream S1 entities in batches
            s1_batch: List[Tuple[str, str, str]] = []
            with open(s1_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None)
                id_idx = header.index("entity_id") if "entity_id" in header else 0
                name_idx = header.index("business_name") if "business_name" in header else 1
                addr_idx = header.index("business_address") if "business_address" in header else 2
                c_idx = header.index("country") if "country" in header else 3

                pbar = tqdm(total=total_country_s1, desc=f"Predicting {country}")
                for row in reader:
                    if len(row) <= c_idx:
                        continue
                    if normalize_country(row[c_idx]) == country:
                        s1_batch.append((
                            row[id_idx],
                            row[name_idx] if len(row) > name_idx else "",
                            row[addr_idx] if len(row) > addr_idx else "",
                        ))

                        if len(s1_batch) >= batch_size:
                            self._process_inference_batch(
                                batch=s1_batch,
                                country_blocker=country_blocker,
                                feature_cols=feature_cols,
                                conn=conn,
                            )
                            pbar.update(len(s1_batch))
                            total_s1_processed += len(s1_batch)
                            s1_batch = []

                if s1_batch:
                    self._process_inference_batch(
                        batch=s1_batch,
                        country_blocker=country_blocker,
                        feature_cols=feature_cols,
                        conn=conn,
                    )
                    pbar.update(len(s1_batch))
                    total_s1_processed += len(s1_batch)
                    s1_batch = []

                pbar.close()

            # Free memory for this country partition
            country_blocker.clear()
            del country_blocker
            gc.collect()

        conn.commit()

        # Export results in the exact original row order of test_source1.tsv
        print(f"\nWriting final output files in exact test_source1.tsv order...")
        print(f"Reading buffered predictions from SQLite...")
        cursor = conn.cursor()
        cursor.execute("SELECT source1_entity_id, candidates, matches FROM results")
        results_map = {}
        while True:
            rows = cursor.fetchmany(100000)
            if not rows:
                break
            for s1_id, cands, matches in rows:
                results_map[s1_id] = (cands, matches)

        conn.close()
        if temp_db_path.exists():
            temp_db_path.unlink()

        print(f"Writing {cand_out_path} and {match_out_path}...")
        written_count = 0
        with open(s1_path, "r", encoding="utf-8", errors="ignore") as f_in, \
             open(cand_out_path, "w", encoding="utf-8") as f_cand, \
             open(match_out_path, "w", encoding="utf-8") as f_match:

            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

            reader = csv.reader(f_in, delimiter="\t")
            header = next(reader, None)
            id_idx = header.index("entity_id") if "entity_id" in header else 0

            for row in reader:
                if not row:
                    continue
                s1_id = row[id_idx].strip()
                cands, matches = results_map.get(s1_id, ("", ""))
                f_cand.write(f"{s1_id}\t{cands}\n")
                f_match.write(f"{s1_id}\t{matches}\n")
                written_count += 1

        del results_map
        gc.collect()

        print(f"Successfully generated {written_count:,} rows in {cand_out_path} and {match_out_path}.")
        return match_out_path, cand_out_path

    def _process_inference_batch(
        self,
        batch: List[Tuple[str, str, str]],
        country_blocker: CountryBlocker,
        feature_cols: List[str],
        conn: sqlite3.Connection,
    ):
        """Process a batch of S1 entities through blocking and scoring, appending to SQLite."""
        batch_pairs = []
        batch_cands: Dict[str, List[str]] = {}
        s1_data_cache = {}

        for s1_id, name, addr in batch:
            clean_name, core_name, sorted_tokens = clean_business_name(name)
            postal_code = extract_postal_code(addr)
            clean_addr = clean_text_basic(addr)
            ngrams = get_char_ngrams(core_name, n=3)
            s1_data = {
                "clean_name": clean_name,
                "core_name": core_name,
                "sorted_tokens": sorted_tokens,
                "clean_addr": clean_addr,
                "postal_code": postal_code,
                "numbers": extract_numbers(clean_addr),
                "ngrams": ngrams,
            }
            s1_data_cache[s1_id] = s1_data

            cand_indices = country_blocker.generate_candidate_indices(
                clean_name=clean_name,
                core_name=core_name,
                sorted_tokens=sorted_tokens,
                postal_code=postal_code,
                ngrams=ngrams,
            )
            cand_eids = [country_blocker.target_ids[idx] for idx in cand_indices]
            batch_cands[s1_id] = cand_eids
            total_cands = len(cand_indices)

            for rank, idx in enumerate(cand_indices):
                t_data = country_blocker.get_target_dict_by_idx(idx)
                feat = extract_pair_features(
                    s1_data=s1_data,
                    target_data=t_data,
                    candidate_rank=rank,
                    total_candidates=total_cands,
                )
                feat["source1_entity_id"] = s1_id
                feat["candidate_entity_id"] = t_data["entity_id"]
                batch_pairs.append(feat)

        # Batch scoring with model
        matches_by_s1: Dict[str, List[str]] = {s1_id: [] for s1_id, _, _ in batch}

        if batch_pairs and self.model.model is not None:
            df_batch = pd.DataFrame(batch_pairs)
            probs = self.model.model.predict_proba(df_batch[feature_cols])[:, 1]
            df_batch["score"] = probs

            grouped = df_batch.groupby("source1_entity_id")
            for s1_id, grp in grouped:
                max_score = grp["score"].max()
                if max_score < self.model.optimal_singleton_thresh:
                    matches_by_s1[s1_id] = []
                else:
                    matched = grp[grp["score"] >= self.model.optimal_threshold]
                    if len(matched) > 0:
                        sorted_matched = matched.sort_values("score", ascending=False)
                        matches_by_s1[s1_id] = sorted_matched["candidate_entity_id"].tolist()[
                            : self.config.max_matches_per_s1
                        ]
                    else:
                        matches_by_s1[s1_id] = []

        # Prepare records for SQLite insertion
        db_records = [
            (s1_id, ",".join(batch_cands.get(s1_id, [])), ",".join(matches_by_s1.get(s1_id, [])))
            for s1_id, _, _ in batch
        ]
        conn.executemany("INSERT INTO results VALUES (?, ?, ?)", db_records)
