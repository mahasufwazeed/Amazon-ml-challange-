"""Data preprocessing and text normalization for Business Entity Resolution.

Supports open-world multilingual text (US, India, France, and any open country)
with Unicode decomposition, legal entity suffix stripping, token reordering,
address component standardization, and missing value resilience.
"""
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# Legal Entity Suffixes (US, India, France, and International)
# -----------------------------------------------------------------------------
LEGAL_SUFFIXES = [
    # Multi-token patterns (check first)
    r"\bpvt\s+ltd\b",
    r"\bprivate\s+limited\b",
    r"\bco\s+ltd\b",
    r"\bco\s+limited\b",
    r"\bltd\s+liab\s+co\b",
    r"\bs\s+a\s+r\s+l\b",
    r"\bs\s+a\s+s\b",
    r"\be\s+u\s+r\s+l\b",
    # India / Commonwealth
    r"\bpvt\b",
    r"\bprivate\b",
    r"\bltd\b",
    r"\blimited\b",
    # US / UK / Global
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bgroup\b",
    r"\bholdings\b",
    r"\benterprises\b",
    r"\bservices\b",
    # France / European
    r"\bsarl\b",
    r"\bsas\b",
    r"\bsasu\b",
    r"\bsa\b",
    r"\bsnc\b",
    r"\beurl\b",
    r"\bgie\b",
    r"\bsci\b",
    r"\bsca\b",
    r"\bgmbh\b",
    r"\bag\b",
]

# Compile legal suffix regex pattern
LEGAL_SUFFIX_REGEX = re.compile(
    r"(?:(?:\s+|-|,)+(?:" + "|".join(LEGAL_SUFFIXES) + r"))+$",
    flags=re.IGNORECASE,
)

# Address abbreviation mappings
ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bhwy\b": "highway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bopp\b": "opposite",
    r"\bnear\b": "near",
    r"\bext\b": "extension",
    r"\bno\b": "number",
}

COMPILED_ADDR_ABBR = [
    (re.compile(pattern, re.IGNORECASE), repl)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items()
]

# Postal code patterns
POSTAL_CODE_REGEX = re.compile(r"\b(\d{5,6})\b")


def strip_accents(text: str) -> str:
    """Normalize Unicode characters (NFKD) and strip combining accents.
    
    Crucial for French entities (e.g., 'Société' -> 'societe', 'Hôtel' -> 'hotel').
    """
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def clean_text_basic(text: Optional[str]) -> str:
    """Basic text sanitization: lowercasing, accent stripping, ampersand handling."""
    if text is None or not isinstance(text, str):
        return ""
    # Unicode decomposition
    text = strip_accents(text.lower())
    # Expand ampersands
    text = re.sub(r"&", " and ", text)
    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Normalize multiple whitespaces
    return " ".join(text.split())


def clean_business_name(name: Optional[str]) -> Tuple[str, str, str]:
    """Process business name into:
    
    1. clean_name: basic sanitized full name
    2. core_name: legal suffixes stripped (e.g. 'Amazon India Pvt Ltd' -> 'amazon india')
    3. sorted_tokens: words alphabetized to handle transpositions
    """
    clean = clean_text_basic(name)
    if not clean:
        return "", "", ""

    # Strip legal suffix from the end
    core = LEGAL_SUFFIX_REGEX.sub("", clean).strip()
    if not core:
        core = clean

    # Token sorting for permutation invariance
    tokens = [t for t in core.split() if len(t) > 0]
    sorted_tokens = " ".join(sorted(tokens))

    return clean, core, sorted_tokens


def extract_postal_code(address: Optional[str]) -> str:
    """Extract standard 5-digit (US, France) or 6-digit (India) postal code."""
    if not address or not isinstance(address, str):
        return ""
    match = POSTAL_CODE_REGEX.search(address)
    return match.group(1) if match else ""


def extract_numbers(address: Optional[str]) -> Set[str]:
    """Extract all distinct number sequences from address (e.g. building/plot/street #)."""
    if not address or not isinstance(address, str):
        return set()
    return set(re.findall(r"\b\d+\b", address))


def clean_address(address: Optional[str]) -> Tuple[str, str, Set[str]]:
    """Process business address into:
    
    1. clean_addr: expanded abbreviations and sanitized address
    2. postal_code: 5 or 6 digit postal code
    3. numbers: set of numeric tokens present
    """
    if address is None or not isinstance(address, str):
        return "", "", set()

    clean = clean_text_basic(address)
    for pattern, repl in COMPILED_ADDR_ABBR:
        clean = pattern.sub(repl, clean)
    clean = " ".join(clean.split())

    postal_code = extract_postal_code(address)
    numbers = extract_numbers(clean)

    return clean, postal_code, numbers


def normalize_country(country: Optional[str]) -> str:
    """Standardize country label as an open string (case-insensitive, trimmed).
    
    Guarantees no hardcoded limitation to {US, India} so that France and
    any future country are supported seamlessly.
    """
    if country is None or not isinstance(country, str):
        return "UNKNOWN"
    norm = strip_accents(country.strip().upper())
    return norm if norm else "UNKNOWN"
