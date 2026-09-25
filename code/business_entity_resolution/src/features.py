"""Feature extraction module for candidate entity pairs.

Computes multi-dimensional name, address, and token similarity features,
resilient to missing addresses and source discrepancies.
"""
from typing import Any, Dict, List, Optional, Set
import difflib

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from .preprocessing import (
    clean_business_name,
    clean_address,
    extract_postal_code,
    extract_numbers,
    clean_text_basic,
)
from .blocking import get_char_ngrams, jaccard_similarity


def compute_string_similarity(s1: str, s2: str) -> Dict[str, float]:
    """Compute string similarity metrics between two strings."""
    if not s1 or not s2:
        return {
            "levenshtein_ratio": 0.0,
            "token_sort_ratio": 0.0,
            "token_set_ratio": 0.0,
            "partial_ratio": 0.0,
        }

    if RAPIDFUZZ_AVAILABLE:
        return {
            "levenshtein_ratio": fuzz.ratio(s1, s2) / 100.0,
            "token_sort_ratio": fuzz.token_sort_ratio(s1, s2) / 100.0,
            "token_set_ratio": fuzz.token_set_ratio(s1, s2) / 100.0,
            "partial_ratio": fuzz.partial_ratio(s1, s2) / 100.0,
        }
    else:
        # Fallback using difflib
        ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
        tokens1 = " ".join(sorted(s1.split()))
        tokens2 = " ".join(sorted(s2.split()))
        sort_ratio = difflib.SequenceMatcher(None, tokens1, tokens2).ratio()
        return {
            "levenshtein_ratio": ratio,
            "token_sort_ratio": sort_ratio,
            "token_set_ratio": sort_ratio,
            "partial_ratio": ratio,
        }


def extract_pair_features(
    s1_data: Dict[str, Any],
    target_data: Dict[str, Any],
    candidate_rank: int = 0,
    total_candidates: int = 1,
) -> Dict[str, float]:
    """Extract a comprehensive feature vector for a pair (Source 1, Candidate)."""
    s1_clean_name = s1_data.get("clean_name", "")
    s1_core_name = s1_data.get("core_name", "")
    s1_sorted_tokens = s1_data.get("sorted_tokens", "")
    s1_addr = s1_data.get("clean_addr", "")
    s1_postal = s1_data.get("postal_code", "")
    s1_numbers = s1_data.get("numbers", set())
    s1_ngrams = s1_data.get("ngrams", set())

    t_clean_name = target_data.get("clean_name", "")
    t_core_name = target_data.get("core_name", "")
    t_sorted_tokens = target_data.get("sorted_tokens", "")
    t_addr = target_data.get("clean_addr", "")
    t_postal = target_data.get("postal_code", "")
    t_numbers = target_data.get("numbers", set())
    t_ngrams = target_data.get("ngrams", set())
    target_id = target_data.get("entity_id", "")

    # -------------------------------------------------------------------------
    # 1. Name Features
    # -------------------------------------------------------------------------
    exact_name_match = 1.0 if s1_clean_name and s1_clean_name == t_clean_name else 0.0
    exact_core_match = 1.0 if s1_core_name and s1_core_name == t_core_name else 0.0
    exact_sorted_match = 1.0 if s1_sorted_tokens and s1_sorted_tokens == t_sorted_tokens else 0.0

    name_sims = compute_string_similarity(s1_core_name, t_core_name)

    # Token level Jaccard
    s1_tokens = set(s1_core_name.split())
    t_tokens = set(t_core_name.split())
    token_jaccard = jaccard_similarity(s1_tokens, t_tokens)

    # Character 3-gram Jaccard
    ngram_jaccard = jaccard_similarity(s1_ngrams, t_ngrams)

    # First word match
    first_w_s1 = s1_core_name.split()[0] if s1_core_name else ""
    first_w_t = t_core_name.split()[0] if t_core_name else ""
    first_word_match = 1.0 if first_w_s1 and first_w_s1 == first_w_t else 0.0

    # Length diff ratio
    len1 = len(s1_core_name)
    len2 = len(t_core_name)
    max_len = max(len1, len2, 1)
    len_diff_ratio = abs(len1 - len2) / max_len

    # -------------------------------------------------------------------------
    # 2. Address Features (Handles missing values safely)
    # -------------------------------------------------------------------------
    s1_has_addr = 1.0 if s1_addr else 0.0
    t_has_addr = 1.0 if t_addr else 0.0
    both_have_addr = 1.0 if (s1_has_addr and t_has_addr) else 0.0

    if both_have_addr:
        addr_sims = compute_string_similarity(s1_addr, t_addr)
        addr_lev_ratio = addr_sims["levenshtein_ratio"]
        addr_token_sort = addr_sims["token_sort_ratio"]

        addr_s1_toks = set(s1_addr.split())
        addr_t_toks = set(t_addr.split())
        addr_token_jaccard = jaccard_similarity(addr_s1_toks, addr_t_toks)
    else:
        addr_lev_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_jaccard = 0.0

    # Postal code match logic: +1 if match, -1 if mismatch, 0 if either missing
    if s1_postal and t_postal:
        postal_match = 1.0 if s1_postal == t_postal else -1.0
    else:
        postal_match = 0.0

    # Number overlap (house/building numbers)
    common_numbers = len(s1_numbers & t_numbers)
    number_jaccard = jaccard_similarity(s1_numbers, t_numbers)

    # -------------------------------------------------------------------------
    # 3. Source & Rank Meta Features
    # -------------------------------------------------------------------------
    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    return {
        "exact_name_match": exact_name_match,
        "exact_core_match": exact_core_match,
        "exact_sorted_match": exact_sorted_match,
        "levenshtein_ratio": name_sims["levenshtein_ratio"],
        "token_sort_ratio": name_sims["token_sort_ratio"],
        "token_set_ratio": name_sims["token_set_ratio"],
        "partial_ratio": name_sims["partial_ratio"],
        "token_jaccard": token_jaccard,
        "ngram_jaccard": ngram_jaccard,
        "first_word_match": first_word_match,
        "len_diff_ratio": len_diff_ratio,
        "s1_has_addr": s1_has_addr,
        "t_has_addr": t_has_addr,
        "both_have_addr": both_have_addr,
        "addr_lev_ratio": addr_lev_ratio,
        "addr_token_sort": addr_token_sort,
        "addr_token_jaccard": addr_token_jaccard,
        "postal_match": postal_match,
        "common_numbers": float(common_numbers),
        "number_jaccard": number_jaccard,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "candidate_rank": float(candidate_rank),
        "total_candidates": float(total_candidates),
    }
