"""Multi-pass blocking and candidate generation for Business Entity Resolution.

Scales across millions of records using country partitioning, multi-key hash indexing,
and character n-gram inverted index fallback.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from tqdm import tqdm

from .preprocessing import clean_business_name, clean_address, normalize_country


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate set of character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class MultiIndexBlocker:
    """Multi-pass hash blocker and inverted index for candidates.
    
    Indexes target records (Source 2 and Source 3) partitioned by country.
    Queries using Source 1 records to retrieve top-K high-recall candidates.
    """

    def __init__(self, max_candidates_per_s1: int = 20, min_ngram_overlap: float = 0.35):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_ngram_overlap = min_ngram_overlap

        # country -> index key -> list of entity_ids
        self.exact_name_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.core_name_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.sorted_tokens_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.token_prefix_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.postal_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Store preprocessed record metadata for target records (S2, S3)
        # entity_id -> {clean_name, core_name, sorted_tokens, postal_code, ngrams}
        self.target_records: Dict[str, dict] = {}

    def index_target_records(self, df_targets: pd.DataFrame, source_label: str = "S2/S3"):
        """Index Source 2 and Source 3 records."""
        print(f"Indexing {len(df_targets):,} records from {source_label}...")

        for row in tqdm(df_targets.itertuples(), total=len(df_targets), desc=f"Indexing {source_label}"):
            eid = row.entity_id
            country = normalize_country(getattr(row, "country", None))
            name = getattr(row, "business_name", "")
            address = getattr(row, "business_address", "")

            clean_name, core_name, sorted_tokens = clean_business_name(name)
            clean_addr, postal_code, numbers = clean_address(address)
            ngrams = get_char_ngrams(core_name, n=3)

            # Store record features
            self.target_records[eid] = {
                "clean_name": clean_name,
                "core_name": core_name,
                "sorted_tokens": sorted_tokens,
                "postal_code": postal_code,
                "numbers": numbers,
                "ngrams": ngrams,
                "country": country,
            }

            # 1. Exact Clean Name Key
            if clean_name:
                self.exact_name_index[country][clean_name].append(eid)

            # 2. Core Name Key (legal suffix stripped)
            if core_name and core_name != clean_name:
                self.core_name_index[country][core_name].append(eid)

            # 3. Sorted Tokens Key (permutation invariant)
            if sorted_tokens:
                self.sorted_tokens_index[country][sorted_tokens].append(eid)

            # 4. First 2 tokens key (for multi-word variations)
            tokens = core_name.split()
            if len(tokens) >= 2:
                prefix_key = " ".join(tokens[:2])
                self.token_prefix_index[country][prefix_key].append(eid)

            # 5. First token + Postal code key
            if tokens and postal_code:
                postal_key = f"{tokens[0]}_{postal_code}"
                self.postal_token_index[country][postal_key].append(eid)

    def generate_candidates_for_record(
        self,
        country: str,
        clean_name: str,
        core_name: str,
        sorted_tokens: str,
        postal_code: str,
        ngrams: Set[str],
    ) -> List[str]:
        """Query multi-index to find best candidate matches within the same country."""
        country_norm = normalize_country(country)
        candidate_scores: Dict[str, float] = defaultdict(float)

        # 1. Exact Name match (high confidence)
        if clean_name in self.exact_name_index[country_norm]:
            for cid in self.exact_name_index[country_norm][clean_name]:
                candidate_scores[cid] = max(candidate_scores[cid], 1.0)

        # 2. Core Name match
        if core_name in self.core_name_index[country_norm]:
            for cid in self.core_name_index[country_norm][core_name]:
                candidate_scores[cid] = max(candidate_scores[cid], 0.95)

        # 3. Sorted tokens match
        if sorted_tokens in self.sorted_tokens_index[country_norm]:
            for cid in self.sorted_tokens_index[country_norm][sorted_tokens]:
                candidate_scores[cid] = max(candidate_scores[cid], 0.90)

        # 4. Token prefix match
        tokens = core_name.split()
        if len(tokens) >= 2:
            prefix_key = " ".join(tokens[:2])
            if prefix_key in self.token_prefix_index[country_norm]:
                for cid in self.token_prefix_index[country_norm][prefix_key][:50]:
                    rec = self.target_records[cid]
                    sim = jaccard_similarity(ngrams, rec["ngrams"])
                    if sim >= self.min_ngram_overlap:
                        candidate_scores[cid] = max(candidate_scores[cid], 0.70 + 0.20 * sim)

        # 5. Postal token match
        if tokens and postal_code:
            postal_key = f"{tokens[0]}_{postal_code}"
            if postal_key in self.postal_token_index[country_norm]:
                for cid in self.postal_token_index[country_norm][postal_key][:30]:
                    rec = self.target_records[cid]
                    sim = jaccard_similarity(ngrams, rec["ngrams"])
                    candidate_scores[cid] = max(candidate_scores[cid], 0.65 + 0.25 * sim)

        # Bonus for postal code match
        if postal_code:
            for cid in list(candidate_scores.keys()):
                if self.target_records[cid]["postal_code"] == postal_code:
                    candidate_scores[cid] += 0.05

        # Sort candidates descending by heuristic score
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Cap at max_candidates_per_s1
        return [cid for cid, score in sorted_candidates[: self.max_candidates_per_s1]]

    def block_s1_dataframe(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        """Run candidate generation for all S1 entities."""
        results: Dict[str, List[str]] = {}

        for row in tqdm(df_s1.itertuples(), total=len(df_s1), desc="Blocking Source 1"):
            eid = row.entity_id
            country = getattr(row, "country", "")
            name = getattr(row, "business_name", "")
            address = getattr(row, "business_address", "")

            clean_name, core_name, sorted_tokens = clean_business_name(name)
            clean_addr, postal_code, numbers = clean_address(address)
            ngrams = get_char_ngrams(core_name, n=3)

            candidates = self.generate_candidates_for_record(
                country=country,
                clean_name=clean_name,
                core_name=core_name,
                sorted_tokens=sorted_tokens,
                postal_code=postal_code,
                ngrams=ngrams,
            )
            results[eid] = candidates

        return results
