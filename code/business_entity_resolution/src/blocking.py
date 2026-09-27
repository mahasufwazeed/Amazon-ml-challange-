"""Multi-pass blocking and candidate generation for Business Entity Resolution.

Scales across millions of records using country partitioning, multi-key hash indexing,
and character n-gram inverted index fallback.
"""
from collections import defaultdict
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from tqdm import tqdm

from .preprocessing import (
    clean_business_name,
    clean_address,
    normalize_country,
    extract_postal_code,
    extract_numbers,
    clean_text_basic,
)


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


class CountryBlocker:
    """Low-memory, high-throughput inverted index blocker for a single country partition.
    
    Stores records in flat parallel arrays and caps inverted index postings for common tokens,
    keeping memory strictly bounded (< 1-2 GB per country).
    """

    def __init__(self, country: str, max_candidates_per_s1: int = 20, min_ngram_overlap: float = 0.30):
        self.country = normalize_country(country)
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_ngram_overlap = min_ngram_overlap

        # Parallel arrays to avoid dict-per-record overhead
        self.target_ids: List[str] = []
        self.target_clean_names: List[str] = []
        self.target_core_names: List[str] = []
        self.target_sorted_tokens: List[str] = []
        self.target_clean_addrs: List[str] = []
        self.target_postal_codes: List[str] = []
        self.target_is_s2: bytearray = bytearray()
        self.eid_to_idx: Dict[str, int] = {}

        # 5 complementary inverted indexes (keys: string, values: lists of int indices)
        self.exact_name_index: Dict[str, List[int]] = defaultdict(list)
        self.core_name_index: Dict[str, List[int]] = defaultdict(list)
        self.sorted_tokens_index: Dict[str, List[int]] = defaultdict(list)
        self.token_prefix_index: Dict[str, List[int]] = defaultdict(list)
        self.postal_token_index: Dict[str, List[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self.target_ids)

    def add_target(self, eid: str, name: str, address: str):
        """Preprocess and index a single target record."""
        clean_name, core_name, sorted_tokens = clean_business_name(name)
        postal_code = extract_postal_code(address)
        clean_addr = clean_text_basic(address)
        is_s2 = 1 if eid.startswith("S2-") else 0

        idx = len(self.target_ids)
        self.target_ids.append(eid)
        self.target_clean_names.append(clean_name)
        self.target_core_names.append(core_name)
        self.target_sorted_tokens.append(sorted_tokens)
        self.target_clean_addrs.append(clean_addr)
        self.target_postal_codes.append(postal_code)
        self.target_is_s2.append(is_s2)
        self.eid_to_idx[eid] = idx

        # 1. Exact clean name key
        if clean_name:
            self.exact_name_index[clean_name].append(idx)

        # 2. Core name key
        if core_name and core_name != clean_name:
            self.core_name_index[core_name].append(idx)

        # 3. Sorted tokens key
        if sorted_tokens and sorted_tokens != clean_name and sorted_tokens != core_name:
            self.sorted_tokens_index[sorted_tokens].append(idx)

        # 4. Token prefix key (first 2 tokens, capped at 40 to prevent stopword memory explosion)
        tokens = core_name.split()
        if len(tokens) >= 2:
            p_key = " ".join(tokens[:2])
            if len(self.token_prefix_index[p_key]) < 40:
                self.token_prefix_index[p_key].append(idx)

        # 5. Token + Postal code key (first token + postal code, capped at 40)
        if tokens and postal_code:
            post_key = f"{tokens[0]}_{postal_code}"
            if len(self.postal_token_index[post_key]) < 40:
                self.postal_token_index[post_key].append(idx)

    def index_from_tsv(self, tsv_path: Path, max_records: Optional[int] = None):
        """Stream TSV line-by-line filtering for this country without allocating DataFrames."""
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header:
                return
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2
            c_idx = header.index("country") if "country" in header else 3

            count = 0
            for row in reader:
                if len(row) > c_idx and normalize_country(row[c_idx]) == self.country:
                    eid = row[id_idx]
                    name = row[name_idx] if len(row) > name_idx else ""
                    addr = row[addr_idx] if len(row) > addr_idx else ""
                    self.add_target(eid, name, addr)
                    count += 1
                    if max_records and count >= max_records:
                        break

    def get_target_dict_by_idx(self, idx: int) -> dict:
        """Generate full record dictionary for candidate featurization on demand."""
        eid = self.target_ids[idx]
        core_name = self.target_core_names[idx]
        clean_addr = self.target_clean_addrs[idx]
        return {
            "entity_id": eid,
            "clean_name": self.target_clean_names[idx],
            "core_name": core_name,
            "sorted_tokens": self.target_sorted_tokens[idx],
            "clean_addr": clean_addr,
            "postal_code": self.target_postal_codes[idx],
            "numbers": extract_numbers(clean_addr),
            "ngrams": get_char_ngrams(core_name, n=3),
            "country": self.country,
            "is_s2": self.target_is_s2[idx] == 1,
        }

    def get_target_dict_by_eid(self, eid: str) -> Optional[dict]:
        idx = self.eid_to_idx.get(eid)
        if idx is None:
            return None
        return self.get_target_dict_by_idx(idx)

    def generate_candidate_indices(
        self,
        clean_name: str,
        core_name: str,
        sorted_tokens: str,
        postal_code: str,
        ngrams: Set[str],
    ) -> List[int]:
        """Query multi-index to find best candidate target indices within this country."""
        candidate_scores: Dict[int, float] = defaultdict(float)

        # 1. Exact Name match (high confidence)
        if clean_name in self.exact_name_index:
            for idx in self.exact_name_index[clean_name]:
                candidate_scores[idx] = max(candidate_scores[idx], 1.0)

        # 2. Core Name match
        if core_name in self.core_name_index:
            for idx in self.core_name_index[core_name]:
                candidate_scores[idx] = max(candidate_scores[idx], 0.95)

        # 3. Sorted tokens match
        if sorted_tokens in self.sorted_tokens_index:
            for idx in self.sorted_tokens_index[sorted_tokens]:
                candidate_scores[idx] = max(candidate_scores[idx], 0.90)

        # 4. Token prefix match
        tokens = core_name.split()
        if len(tokens) >= 2:
            prefix_key = " ".join(tokens[:2])
            if prefix_key in self.token_prefix_index:
                for idx in self.token_prefix_index[prefix_key]:
                    cand_ngrams = get_char_ngrams(self.target_core_names[idx], n=3)
                    sim = jaccard_similarity(ngrams, cand_ngrams)
                    if sim >= self.min_ngram_overlap:
                        candidate_scores[idx] = max(candidate_scores[idx], 0.70 + 0.20 * sim)

        # 5. Postal token match
        if tokens and postal_code:
            postal_key = f"{tokens[0]}_{postal_code}"
            if postal_key in self.postal_token_index:
                for idx in self.postal_token_index[postal_key]:
                    cand_ngrams = get_char_ngrams(self.target_core_names[idx], n=3)
                    sim = jaccard_similarity(ngrams, cand_ngrams)
                    candidate_scores[idx] = max(candidate_scores[idx], 0.65 + 0.25 * sim)

        # Bonus for postal code match
        if postal_code:
            for idx in list(candidate_scores.keys()):
                if self.target_postal_codes[idx] == postal_code:
                    candidate_scores[idx] += 0.05

        # Sort candidates descending by heuristic score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_candidates[: self.max_candidates_per_s1]]

    def generate_candidate_eids(
        self,
        clean_name: str,
        core_name: str,
        sorted_tokens: str,
        postal_code: str,
        ngrams: Set[str],
    ) -> List[str]:
        cand_indices = self.generate_candidate_indices(
            clean_name=clean_name,
            core_name=core_name,
            sorted_tokens=sorted_tokens,
            postal_code=postal_code,
            ngrams=ngrams,
        )
        return [self.target_ids[idx] for idx in cand_indices]

    def clear(self):
        """Free all internal data structures to immediately release RAM."""
        self.target_ids.clear()
        self.target_clean_names.clear()
        self.target_core_names.clear()
        self.target_sorted_tokens.clear()
        self.target_clean_addrs.clear()
        self.target_postal_codes.clear()
        self.target_is_s2.clear()
        self.eid_to_idx.clear()
        self.exact_name_index.clear()
        self.core_name_index.clear()
        self.sorted_tokens_index.clear()
        self.token_prefix_index.clear()
        self.postal_token_index.clear()


class MultiIndexBlocker:
    """Multi-pass hash blocker and inverted index for candidates across countries."""

    def __init__(self, max_candidates_per_s1: int = 20, min_ngram_overlap: float = 0.35):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_ngram_overlap = min_ngram_overlap
        self.country_blockers: Dict[str, CountryBlocker] = {}
        self.eid_to_country: Dict[str, str] = {}

    def get_country_blocker(self, country: str) -> CountryBlocker:
        c_norm = normalize_country(country)
        if c_norm not in self.country_blockers:
            self.country_blockers[c_norm] = CountryBlocker(
                country=c_norm,
                max_candidates_per_s1=self.max_candidates_per_s1,
                min_ngram_overlap=self.min_ngram_overlap,
            )
        return self.country_blockers[c_norm]

    def index_target_records(self, df_targets: pd.DataFrame, source_label: str = "S2/S3"):
        """Index Source 2 and Source 3 records."""
        print(f"Indexing {len(df_targets):,} records from {source_label}...")
        for row in tqdm(df_targets.itertuples(), total=len(df_targets), desc=f"Indexing {source_label}"):
            eid = row.entity_id
            c_norm = normalize_country(getattr(row, "country", None))
            name = getattr(row, "business_name", "")
            address = getattr(row, "business_address", "")
            blocker = self.get_country_blocker(c_norm)
            blocker.add_target(eid, name, address)
            self.eid_to_country[eid] = c_norm

    def get_target_record_dict(self, eid: str) -> dict:
        c_norm = self.eid_to_country.get(eid)
        if c_norm and c_norm in self.country_blockers:
            d = self.country_blockers[c_norm].get_target_dict_by_eid(eid)
            if d is not None:
                return d
        # Fallback search across all country blockers
        for blocker in self.country_blockers.values():
            d = blocker.get_target_dict_by_eid(eid)
            if d is not None:
                return d
        raise KeyError(f"Entity ID {eid} not found in blocker target records")

    @property
    def target_records(self) -> dict:
        """Compatibility property for code checking `cid in blocker.target_records`."""
        return self.eid_to_country

    def generate_candidates_for_record(
        self,
        country: str,
        clean_name: str,
        core_name: str,
        sorted_tokens: str,
        postal_code: str,
        ngrams: Set[str],
    ) -> List[str]:
        c_norm = normalize_country(country)
        if c_norm not in self.country_blockers:
            return []
        return self.country_blockers[c_norm].generate_candidate_eids(
            clean_name=clean_name,
            core_name=core_name,
            sorted_tokens=sorted_tokens,
            postal_code=postal_code,
            ngrams=ngrams,
        )

    def block_s1_dataframe(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
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
