# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Memory-Optimized](https://img.shields.io/badge/RAM-16GB%20Optimized%20(<2.5GB%20Peak)-brightgreen.svg)]()
[![Evaluation](https://img.shields.io/badge/Evaluation-Macro%20F0.5-orange.svg)]()

Production-grade, memory-efficient Entity Resolution (ER) system built for the **Amazon ML Challenge 2026**. 
Given business records across three noisy, heterogeneous data sources (`Source 1`, `Source 2`, and `Source 3`), the system links and matches records across sources to identify real-world business entities.

---

## 1. System Optimization & Crash Resolution (for 16 GB RAM Laptops)

### The Problem (Why the Previous Run Crashed)
The raw dataset contains over **12.5 Million records** across three sources (~2.5 GB of raw TSV text):
- **Source 1**: ~2.2M training entities, ~1.73M test entities
- **Source 2**: ~5.03M training records, ~4.88M test records
- **Source 3**: ~5.28M training records, ~5.08M test records

The unoptimized implementation attempted to:
1. Load all 10.3M target records at once into memory via `pd.read_csv` (~4.5 GB).
2. Precompute and store `frozenset` objects for character 3-grams and address numbers for all 10.3M records in memory (>10 GB).
3. Build 5 monolithic inverted hash tables with `(country, token)` tuple keys across all records simultaneously (~6 GB).
4. Featurize 25M–35M candidate pairs into one gigantic in-memory DataFrame (>15 GB).

On a 16 GB RAM laptop (~7.2 GB available memory), this exceeded **30 GB of RAM**, forcing Windows into SSD pagefile thrashing, freezing the system, and crashing with an Out-Of-Memory (OOM) error.

---

### The Solution (Memory-Bounded Streaming Architecture)

We refactored the entire blocking, feature extraction, and inference pipeline to operate under a strict **<2.5 GB RAM ceiling**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Test / Train TSV Files                          │
│               (Streamed line-by-line via C csv.reader)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Stage 1: Country Partitioning Strategy                 │
│      - Process ONE country partition at a time:                        │
│          1. France (1.4M targets, ~400 MB RAM)                         │
│          2. US (3.8M targets, ~1.1 GB RAM)                             │
│          3. India (4.7M targets, ~1.4 GB RAM)                          │
│      - Immediate garbage collection (`del; gc.collect()`) per country │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 2: CountryBlocker Inverted Indexing                │
│      - Flat parallel arrays for target storage (no dict per record)    │
│      - Inverted indexes: Exact Name, Core Name, Sorted Tokens,         │
│        2-Token Prefix, First Token + Postal Code                       │
│      - Frequency capping on common prefixes (<=40 candidates)          │
│      - On-demand N-gram and number extraction (eliminated 8 GB RAM)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 3: Streaming Batch Feature & Scoring               │
│      - Source 1 entities processed in batches of 10,000                │
│      - Featurize only retrieved candidates (~60k pairs/batch = ~19 MB) │
│      - RapidFuzz similarity evaluation (87,000 pairs/sec)              │
│      - LightGBM predict_proba with calibrated threshold & singleton    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Stage 4: Zero-RAM Disk Buffer & Exact Ordering             │
│      - Stream predictions into temporary SQLite disk buffer            │
│      - Single-pass export into exact test_source1.tsv row order        │
│      - matching_results.tsv & candidate_pairs.tsv generated            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Stage 5: Exhaustive Submission Compliance Audit            │
│      - Validate exact entity count, headers, subset condition          │
│      - 100% compliant with competition validator (PASS, exit 0)        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Performance Benchmarks Achieved:
| Optimization | Before | After | Improvement |
| :--- | :---: | :---: | :---: |
| **ASCII Text Normalization** | 2.76s / 1M strings | **0.08s / 1M strings** | **35x Faster** |
| **Target Indexing Speed** | ~17–20 mins (OOM) | **2.04s / 100k records** | **~10x Faster** |
| **Peak RAM Usage** | >30 GB (Crashed) | **<2.5 GB** | **>90% Memory Reduction** |
| **Feature Extraction Speed** | Pure Python loop | **87,000 pairs/sec** | High-throughput |
| **Candidate Blocking Speed** | Multi-dict scan | **0.15s / 10,000 entities** | Sub-second |

---

## 2. Repository Structure

```
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py           # Configuration, paths, thresholds
│       │   ├── preprocessing.py    # 35x faster Unicode & legal suffix cleaning
│       │   ├── blocking.py         # CountryBlocker & MultiIndexBlocker (<2.5 GB RAM)
│       │   ├── features.py         # Pairwise 24-dimensional feature extraction
│       │   ├── metrics.py          # Competition Macro F_0.5 metric evaluation
│       │   ├── model.py            # LightGBM classifier & threshold calibrator
│       │   ├── pipeline.py         # Streaming country-partitioned batch pipeline
│       │   └── validate.py         # Format and integrity submission validator
│       ├── requirements.txt        # Pinned Python package dependencies
│       └── README.md               # Code package documentation
├── dataset/
│   ├── train/                      # Training TSV files (source1, source2, source3, ground truth)
│   ├── test/                       # Test TSV files (source1, source2, source3)
│   ├── utils/
│   │   └── validate_submission.py  # Official competition validator
│   └── README.md                   # Problem statement & competition rules
├── models/                         # Saved trained model artifacts (er_lgbm_model.joblib)
├── output/                         # Generated competition output TSVs
│   ├── candidate_pairs.tsv         # Blocking candidate pairs
│   └── matching_results.tsv        # Final scored entity matches
├── scripts/
│   └── generate_synthetic_test_data.py
├── tests/
│   └── test_pipeline.py            # Unit & regression test suite (100% PASS)
├── Documentation_template.md       # Methodology documentation
├── package_submission.py           # Submission zip packager
├── run_pipeline.py                 # CLI entry point to run train, predict, validate
└── README.md                       # This comprehensive documentation
```

---

## 3. Quick Start & Execution

### 1. Environment Requirements
- Python 3.10 or 3.12 (64-bit)
- Packages: `lightgbm`, `rapidfuzz`, `pandas`, `numpy`, `scikit-learn`, `tqdm`, `joblib`

Verify dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Verify Pipeline Integrity
Run the unit test suite:
```bash
pytest tests/
```
Expected output:
```
============================== 7 passed in 0.50s ==============================
```

### 3. Run End-to-End Pipeline
To run training, prediction, and validation in one go:
```bash
python run_pipeline.py --train --predict --validate --sample-size 40000
```

#### What this command does:
1. **`--train`**: Samples 40,000 balanced entities (20k US, 20k India) from training data, indexes relevant target records, trains the LightGBM matcher, tunes decision thresholds for the Macro $F_{0.5}$ metric, and saves the trained model to `models/er_lgbm_model.joblib`.
2. **`--predict`**: Automatically discovers test countries (`France`, `US`, `India`), indexes target records country-by-country, processes all 1.73M test entities in streaming batches of 10,000, and writes:
   - `output/candidate_pairs.tsv`
   - `output/matching_results.tsv`
3. **`--validate`**: Verifies that both output files strictly satisfy all competition rules (valid headers, exact test S1 row count, correct ID prefixes, no self-matches, no duplicates, subset condition).

---

## 4. Packaging the Final Submission

Once the output files are generated and validated, package the complete submission zip:
```bash
python package_submission.py
```
This produces `<team_name>_submission.zip` containing:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `code/business_entity_resolution/` (complete source code & instructions)
- `Documentation_template.md` (methodology write-up)
