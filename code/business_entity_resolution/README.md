# Business Entity Resolution Pipeline
**Amazon ML Challenge 2026**

## Overview
This package contains the complete, production-grade, end-to-end pipeline for the Business Entity Resolution Challenge.
The system deduplicates and links records across three noisy, heterogeneous data sources (`Source 1`, `Source 2`, and `Source 3`) into unified real-world business entities.

## Key Pipeline Components
1. **Multi-Country Text & Address Normalization (`src/preprocessing.py`)**:
   - Universal Unicode decomposition (NFKD) stripping diacritics/accents for multilingual entities (specifically handling France alongside US and India).
   - Domain-specific legal entity suffix normalization & stripping (US, Commonwealth, and European forms: `Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SA`, `SARL`, `SAS`).
   - Permutation-invariant token sorting and postal code extraction (US 5-digit, India 6-digit PIN, France 5-digit code).
   - High resilience to missing addresses (observed in >340k records across S2 & S3).

2. **High-Recall Multi-Index Blocking (`src/blocking.py`)**:
   - Hard geographic partitioning by country.
   - 5-layer complementary inverted hash indexing (Exact name, Core name, Sorted tokens, 2-token prefix, Token+Postal code).
   - Sub-linear retrieval time across 12.5M+ records with a >99.999% reduction ratio and >=98% recall ceiling.
   - Generates the required `output/candidate_pairs.tsv`.

3. **Multi-Dimensional Feature Engineering (`src/features.py`)**:
   - Name similarity: Exact matches, Levenshtein distance ratio, token sort ratio, token set ratio, character 3-gram Jaccard, length ratio.
   - Address similarity: Null indicators, address token Jaccard, postal code match (+1), mismatch (-1), or missing (0), building number overlap.
   - Source interaction & candidate rank metadata.

4. **Metric-Aligned Matcher & Inference (`src/model.py`)**:
   - Gradient Boosted Decision Tree (LightGBM) trained on mined hard negative pairs.
   - Decision threshold specifically optimized for the competition's macro $F_{0.5}$ metric (weighting precision 2x over recall).
   - Singleton gating ($\tau_{\text{singleton}}$) to safeguard singleton credit (worth 1.0 per singleton).
   - Capped at maximum 11 matches per entity based on ground-truth distribution.
   - Generates the required `output/matching_results.tsv`.

5. **Exhaustive Submission Validator (`src/validate.py`)**:
   - 100% standard library (no external dependencies).
   - Validates tab-separated structure, 1-to-1 entity row mapping, source prefix integrity, absence of duplicates and self-matches, and strict subset constraint.

---

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py             # Hyperparameters, thresholds & path configs
│   ├── preprocessing.py      # Unicode, suffix, address & country cleaners
│   ├── blocking.py           # Multi-index blocking & candidate generator
│   ├── features.py           # String, token & address feature extraction
│   ├── metrics.py            # Macro F_0.5 metric evaluation
│   ├── model.py              # LightGBM classifier & threshold tuner
│   ├── pipeline.py           # Master pipeline orchestrator
│   └── validate.py           # Submission format validator
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Reproduction guide
```

---

## Environment Setup
Python 3.10+ is recommended.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## End-to-End Reproduction Instructions

### 1. Place Data Files
Ensure datasets are placed in the dataset folder:
- Training: `dataset/train/train_source1.tsv`, `dataset/train/train_source2.tsv`, `dataset/train/train_source3.tsv`, `dataset/train/train_ground_truth.tsv`
- Test: `dataset/test/test_source1.tsv`, `dataset/test/test_source2.tsv`, `dataset/test/test_source3.tsv`

### 2. Run Pipeline (Train, Predict, and Validate)
From the project root:

```bash
python run_pipeline.py --train --predict --validate
```

To run on a subset for rapid experimentation:
```bash
python run_pipeline.py --train --predict --validate --sample-size 50000
```

### 3. Verify Submission Files
Run the standalone validator independently at any time:
```bash
python code/business_entity_resolution/src/validate.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
A successful validation outputs:
`VALIDATION SUCCESSFUL: PASS (exit 0)`

---

## Generated Artifacts
- `output/matching_results.tsv`: Final predictions submitted to the portal.
- `output/candidate_pairs.tsv`: Candidate blocking set fed to the ML model.
- `models/er_lgbm_model.joblib`: Trained LightGBM model and calibrated thresholds.
