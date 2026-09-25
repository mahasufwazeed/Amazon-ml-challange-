"""Comprehensive Unit & Regression Tests for the Entity Resolution Pipeline."""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.preprocessing import (
    strip_accents,
    clean_business_name,
    clean_address,
    extract_postal_code,
    normalize_country,
)
from src.blocking import MultiIndexBlocker, get_char_ngrams, jaccard_similarity
from src.features import extract_pair_features
from src.metrics import compute_entity_f_beta, evaluate_macro_f_beta
from src.validate import validate_submission_files


def test_preprocessing_french_accents():
    """Verify French characters and diacritics are cleanly normalized."""
    text = "Société Générale d'Assurance à Paris"
    clean, core, sorted_tokens = clean_business_name(text)
    assert "societe generale" in core
    assert "é" not in core and "à" not in core


def test_preprocessing_legal_suffixes():
    """Verify legal entity suffixes are stripped properly across regions."""
    # India
    _, core_in, _ = clean_business_name("Infosys Technologies Pvt Ltd")
    assert core_in == "infosys technologies"

    # US
    _, core_us, _ = clean_business_name("Google LLC")
    assert core_us == "google"

    # France
    _, core_fr, _ = clean_business_name("TotalEnergies SE")
    # Token sorting
    _, _, sorted_tok = clean_business_name("Apex Solutions")
    assert sorted_tok == "apex solutions"
    _, _, sorted_tok_rev = clean_business_name("Solutions Apex")
    assert sorted_tok == sorted_tok_rev


def test_address_and_postal_extraction():
    """Verify postal codes and address normalizations."""
    # US zip
    clean_us, zip_us, nums_us = clean_address("123 Main St, Suite 400, Seattle, WA 98101")
    assert zip_us == "98101"
    assert "street" in clean_us
    assert "suite" in clean_us
    assert "123" in nums_us

    # India PIN
    clean_in, pin_in, nums_in = clean_address("Plot 45, Near SBI ATM, MG Rd, Bangalore 560001")
    assert pin_in == "560001"
    assert "road" in clean_in

    # France postal code
    clean_fr, pin_fr, _ = clean_address("15 Rue de Rivoli, 75001 Paris")
    assert pin_fr == "75001"

    # Missing address
    clean_null, pin_null, nums_null = clean_address(None)
    assert clean_null == ""
    assert pin_null == ""
    assert nums_null == set()


def test_open_country_normalization():
    """Verify country normalization handles open set without restricting to {US, India}."""
    assert normalize_country("US") == "US"
    assert normalize_country("India") == "INDIA"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("Allemagne") == "ALLEMAGNE"
    assert normalize_country(None) == "UNKNOWN"


def test_metric_pdf_example():
    """Verify exact numerical agreement with the problem statement example:
    
    Model predicts: [S2-00047, S2-00193, S3-00812] (3 preds, 2 true positives)
    Ground truth:   [S2-00047, S3-00812] (2 true)
    P = 2/3 = 0.6667, R = 1.0 -> F_0.5 = 0.714
    """
    preds = {"S2-00047", "S2-00193", "S3-00812"}
    gt = {"S2-00047", "S3-00812"}
    f_score, prec, rec = compute_entity_f_beta(preds, gt, beta=0.5)

    assert round(prec, 3) == 0.667
    assert round(rec, 3) == 1.0
    assert round(f_score, 3) == 0.714


def test_metric_singletons():
    """Verify singleton evaluation:
    
    - Correct empty prediction on singleton = 1.0
    - False merge on singleton = 0.0
    """
    # Correct singleton
    f1, _, _ = compute_entity_f_beta(set(), set(), beta=0.5)
    assert f1 == 1.0

    # False positive on singleton
    f0, _, _ = compute_entity_f_beta({"S2-001"}, set(), beta=0.5)
    assert f0 == 0.0

    # Missed match on non-singleton
    f_miss, _, _ = compute_entity_f_beta(set(), {"S2-001"}, beta=0.5)
    assert f_miss == 0.0


def test_feature_extraction_resilience():
    """Verify feature extractor handles both populated and null fields safely."""
    s1_data = {
        "clean_name": "apple store",
        "core_name": "apple store",
        "sorted_tokens": "apple store",
        "clean_addr": "1 infinite loop",
        "postal_code": "95014",
        "numbers": {"1"},
        "ngrams": get_char_ngrams("apple store", n=3),
    }

    # Case A: Complete target
    t_data_complete = {
        "entity_id": "S2-1001",
        "clean_name": "apple store retail",
        "core_name": "apple store",
        "sorted_tokens": "apple store",
        "clean_addr": "1 infinite loop",
        "postal_code": "95014",
        "numbers": {"1"},
        "ngrams": get_char_ngrams("apple store", n=3),
    }
    feats = extract_pair_features(s1_data, t_data_complete, candidate_rank=0, total_candidates=1)
    assert feats["exact_core_match"] == 1.0
    assert feats["postal_match"] == 1.0
    assert feats["is_s2"] == 1.0

    # Case B: Target with missing address (common in S2/S3)
    t_data_missing_addr = {
        "entity_id": "S3-2002",
        "clean_name": "apple store",
        "core_name": "apple store",
        "sorted_tokens": "apple store",
        "clean_addr": "",
        "postal_code": "",
        "numbers": set(),
        "ngrams": get_char_ngrams("apple store", n=3),
    }
    feats_null = extract_pair_features(s1_data, t_data_missing_addr, candidate_rank=1, total_candidates=2)
    assert feats_null["t_has_addr"] == 0.0
    assert feats_null["both_have_addr"] == 0.0
    assert feats_null["postal_match"] == 0.0
    assert feats_null["is_s3"] == 1.0
