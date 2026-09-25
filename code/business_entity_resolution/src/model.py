"""Model training, inference, and threshold calibration for Business Entity Resolution.

Optimized specifically for the macro F_0.5 evaluation metric with singleton gating
and high-precision candidate filtering.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from .config import ERConfig, default_config
from .metrics import evaluate_macro_f_beta, compute_entity_f_beta


class ERClassifier:
    """LightGBM-based Entity Resolution Matcher with metric-aligned inference."""

    def __init__(self, config: Optional[ERConfig] = None):
        self.config = config or default_config
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = []
        self.optimal_threshold: float = self.config.classification_threshold
        self.optimal_singleton_thresh: float = self.config.singleton_threshold

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train LightGBM binary classifier on candidate pairs."""
        self.feature_names = list(X_train.columns)

        min_child = max(2, min(20, len(X_train) // 4))
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.config.random_seed,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1,
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        print("Fitting LightGBM matcher...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None,
        )
        print("Training complete.")

    def calibrate_thresholds(
        self,
        df_candidates: pd.DataFrame,
        ground_truth: Dict[str, List[str]],
        feature_cols: List[str],
    ) -> Tuple[float, float]:
        """Grid search over classification and singleton thresholds to maximize Macro F_0.5."""
        print("Calibrating decision thresholds for Macro F_0.5...")
        X = df_candidates[feature_cols]
        probs = self.model.predict_proba(X)[:, 1]
        df_candidates = df_candidates.copy()
        df_candidates["score"] = probs

        best_score = -1.0
        best_tau = self.optimal_threshold
        best_sing = self.optimal_singleton_thresh

        # Precision-oriented threshold sweep
        candidate_taus = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        candidate_sings = [0.50, 0.55, 0.60, 0.65, 0.70]

        # Group candidates by S1 entity
        grouped = df_candidates.groupby("source1_entity_id")

        for tau in candidate_taus:
            for sing in candidate_sings:
                preds: Dict[str, List[str]] = {}
                for s1_id, group in grouped:
                    max_score = group["score"].max()
                    if max_score < sing:
                        preds[s1_id] = []
                    else:
                        matches = group[group["score"] >= tau]
                        sorted_matches = matches.sort_values(
                            "score", ascending=False
                        )
                        pred_ids = sorted_matches["candidate_entity_id"].tolist()[
                            : self.config.max_matches_per_s1
                        ]
                        preds[s1_id] = pred_ids

                # Evaluate on ground truth
                metrics = evaluate_macro_f_beta(preds, ground_truth, beta=0.5)
                score = metrics["macro_f_0.5"]
                if score > best_score:
                    best_score = score
                    best_tau = tau
                    best_sing = sing

        print(
            f"Optimized Thresholds: Match Threshold={best_tau:.2f}, "
            f"Singleton Threshold={best_sing:.2f} -> Validation Macro F_0.5: {best_score:.4f}"
        )
        self.optimal_threshold = best_tau
        self.optimal_singleton_thresh = best_sing
        return best_tau, best_sing

    def predict_matches(
        self,
        candidate_dict: Dict[str, List[str]],
        feature_df_dict: Optional[Dict[str, pd.DataFrame]] = None,
        feature_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[str]]:
        """Run high-precision inference across all S1 candidates."""
        if feature_matrix is None and feature_df_dict is None:
            raise ValueError("Must provide candidate feature data.")

        # Batch probability prediction
        if feature_matrix is not None:
            if self.model is not None:
                probs = self.model.predict_proba(feature_matrix[self.feature_names])[:, 1]
                feature_matrix = feature_matrix.copy()
                feature_matrix["score"] = probs
            else:
                # Heuristic scoring fallback
                feature_matrix["score"] = (
                    0.5 * feature_matrix["levenshtein_ratio"]
                    + 0.3 * feature_matrix["token_jaccard"]
                    + 0.2 * feature_matrix["exact_core_match"]
                )

            grouped = feature_matrix.groupby("source1_entity_id")
            scores_by_s1 = {s1_id: grp for s1_id, grp in grouped}
        else:
            scores_by_s1 = {}

        results: Dict[str, List[str]] = {}

        for s1_id, candidates in candidate_dict.items():
            if not candidates or s1_id not in scores_by_s1:
                results[s1_id] = []
                continue

            grp = scores_by_s1[s1_id]
            max_score = grp["score"].max()

            # 1. Singleton Guard
            if max_score < self.optimal_singleton_thresh:
                results[s1_id] = []
                continue

            # 2. Match thresholding
            matched = grp[grp["score"] >= self.optimal_threshold]
            if len(matched) == 0:
                # Fallback: if highest confidence is high enough and clearly above runner-up
                results[s1_id] = []
            else:
                sorted_matched = matched.sort_values("score", ascending=False)
                # Cap at max_matches_per_s1
                pred_ids = sorted_matched["candidate_entity_id"].tolist()[
                    : self.config.max_matches_per_s1
                ]
                results[s1_id] = pred_ids

        return results

    def save(self, output_path: Path):
        """Persist model and calibration parameters."""
        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "optimal_threshold": self.optimal_threshold,
            "optimal_singleton_thresh": self.optimal_singleton_thresh,
        }
        joblib.dump(state, output_path)
        print(f"Model saved to {output_path}")

    def load(self, model_path: Path):
        """Load trained model and parameters."""
        state = joblib.load(model_path)
        self.model = state["model"]
        self.feature_names = state["feature_names"]
        self.optimal_threshold = state["optimal_threshold"]
        self.optimal_singleton_thresh = state["optimal_singleton_thresh"]
        print(f"Model loaded from {model_path}")
