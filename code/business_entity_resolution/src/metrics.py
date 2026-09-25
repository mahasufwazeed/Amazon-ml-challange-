"""Evaluation metrics for the Amazon ML Challenge 2026.

Implements the exact competition Macro-Averaged F_0.5 score:
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def compute_entity_f_beta(
    pred_set: Set[str],
    true_set: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute F_beta for a single Source 1 entity.
    
    Returns: (f_score, precision, recall)
    """
    # Case 1: Singleton entity (Ground truth has no matches)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            # Correctly identified singleton
            return 1.0, 1.0, 1.0
        else:
            # False merge on singleton
            return 0.0, 0.0, 0.0

    # Case 2: Entity has true matches
    if len(pred_set) == 0:
        # Missed all matches
        return 0.0, 0.0, 0.0

    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    f_score = numerator / denominator if denominator > 0 else 0.0
    return f_score, precision, recall


def evaluate_macro_f_beta(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Compute Macro-averaged F_beta across all Source 1 entities in evaluation set."""
    f_scores = []
    precisions = []
    recalls = []

    singleton_total = 0
    singleton_correct = 0
    matched_total = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        pred_set = set(predictions.get(s1_id, []))

        is_singleton = len(true_set) == 0
        if is_singleton:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            matched_total += 1

        f_score, prec, rec = compute_entity_f_beta(pred_set, true_set, beta=beta)
        f_scores.append(f_score)
        precisions.append(prec)
        recalls.append(rec)

    macro_f = float(np.mean(f_scores)) if f_scores else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = (
        (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    )

    return {
        f"macro_f_{beta}": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "singleton_count": singleton_total,
        "matched_count": matched_total,
        "total_evaluated": len(ground_truth),
    }
