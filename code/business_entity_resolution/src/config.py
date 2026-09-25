"""Configuration parameters for the Entity Resolution Pipeline.

Adheres strictly to the Amazon ML Challenge 2026 guidelines:
- Evaluation: Macro-averaged F_0.5 score (weights precision 2x over recall)
- Constraint: Open-world country support (US, India, France, etc.)
- Output: matching_results.tsv and candidate_pairs.tsv in output/
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ERConfig:
    # -------------------------------------------------------------------------
    # Directory & File Paths
    # -------------------------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = field(default_factory=lambda: Path("dataset"))
    train_dir: Path = field(default_factory=lambda: Path("dataset/train"))
    test_dir: Path = field(default_factory=lambda: Path("dataset/test"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # File names
    matching_output_filename: str = "matching_results.tsv"
    candidate_output_filename: str = "candidate_pairs.tsv"

    # -------------------------------------------------------------------------
    # Blocking / Candidate Generation
    # -------------------------------------------------------------------------
    # Maximum candidates retained per Source 1 entity to bound downstream complexity
    max_candidates_per_s1: int = 20
    # Minimum shingle length for character n-gram blocking
    ngram_size: int = 3
    # Minimum Jaccard overlap threshold for fuzzy n-gram blocking fallback
    ngram_overlap_threshold: float = 0.30

    # -------------------------------------------------------------------------
    # Model & Metric Optimization (Macro F_0.5)
    # -------------------------------------------------------------------------
    # Beta value for F-beta score (0.5 prioritizes precision over recall)
    beta: float = 0.5
    # Decision threshold for binary classifier (tuned for high precision)
    classification_threshold: float = 0.72
    # Confidence threshold to declare an entity a singleton (no matches)
    # If the top candidate score is below this threshold, output empty list
    singleton_threshold: float = 0.65
    # Historical maximum matches observed for any S1 entity
    max_matches_per_s1: int = 11

    # -------------------------------------------------------------------------
    # Training & Reproducibility
    # -------------------------------------------------------------------------
    random_seed: int = 42
    n_splits: int = 5
    num_leaves: int = 63
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 300

    def resolve_paths(self):
        """Ensure all relative paths are properly anchored to base_dir."""
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        if not self.train_dir.is_absolute():
            self.train_dir = self.base_dir / self.train_dir
        if not self.test_dir.is_absolute():
            self.test_dir = self.base_dir / self.test_dir
        if not self.output_dir.is_absolute():
            self.output_dir = self.base_dir / self.output_dir
        if not self.model_dir.is_absolute():
            self.model_dir = self.base_dir / self.model_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


default_config = ERConfig()
