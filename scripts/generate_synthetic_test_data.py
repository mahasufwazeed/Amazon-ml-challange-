"""Generate synthetic benchmark datasets mirroring the Amazon ML Challenge 2026 schema.

Includes:
- Noise patterns: legal suffix variation, typos, word transpositions, address abbreviations
- Missing addresses (mimicking S2/S3 missing address distribution)
- Singletons (entities with 0 matches)
- Multi-matches (entities with 1 to 5 matches across S2 and S3)
- Country distribution: US and India in Train; US, India, AND France in Test!
"""
import random
from pathlib import Path
import pandas as pd

random.seed(42)

TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Base Entity Templates
# -----------------------------------------------------------------------------
TRAIN_TEMPLATES = [
    # US
    {"name": "Starbucks Coffee Company", "addr": "2401 Utah Ave S, Suite 800, Seattle, WA 98134", "country": "US"},
    {"name": "Microsoft Corporation", "addr": "One Microsoft Way, Redmond, WA 98052", "country": "US"},
    {"name": "Walmart Inc", "addr": "702 SW 8th St, Bentonville, AR 72716", "country": "US"},
    {"name": "The Home Depot Inc", "addr": "2455 Paces Ferry Rd NW, Atlanta, GA 30339", "country": "US"},
    {"name": "Costco Wholesale Corp", "addr": "999 Lake Dr, Issaquah, WA 98027", "country": "US"},
    {"name": "Target Brands LLC", "addr": "1000 Nicollet Mall, Minneapolis, MN 55403", "country": "US"},
    {"name": "Nike Retail Services", "addr": "One Bowerman Dr, Beaverton, OR 97005", "country": "US"},
    {"name": "FedEx Ground Package System", "addr": "1000 FedEx Dr, Moon Twp, PA 15108", "country": "US"},
    {"name": "Singleton US Store", "addr": "456 Desert Rd, Reno, NV 89501", "country": "US"}, # Singleton
    # India
    {"name": "Tata Consultancy Services Ltd", "addr": "TCS House, Raveline St, Fort, Mumbai 400001", "country": "India"},
    {"name": "Infosys Limited", "addr": "Electronics City, Hosur Rd, Bangalore 560100", "country": "India"},
    {"name": "Wipro Enterprises Pvt Ltd", "addr": "Doddakannelli, Sarjapur Rd, Bangalore 560035", "country": "India"},
    {"name": "Reliance Retail Limited", "addr": "Maker Chambers IV, Nariman Point, Mumbai 400021", "country": "India"},
    {"name": "HDFC Bank Ltd", "addr": "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "country": "India"},
    {"name": "State Bank of India", "addr": "State Bank Bhavan, Madame Cama Rd, Nariman Point, Mumbai 400021", "country": "India"},
    {"name": "Singleton India Traders", "addr": "Near Clock Tower, Chandni Chowk, Delhi 110006", "country": "India"}, # Singleton
]

TEST_TEMPLATES = [
    # US
    {"name": "Amazon Services LLC", "addr": "410 Terry Ave N, Seattle, WA 98109", "country": "US"},
    {"name": "Apple Computer Inc", "addr": "One Infinite Loop, Cupertino, CA 95014", "country": "US"},
    {"name": "Alphabet Google LLC", "addr": "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "country": "US"},
    {"name": "Tesla Motors Inc", "addr": "3500 Deer Creek Rd, Palo Alto, CA 94304", "country": "US"},
    {"name": "Singleton Test US Firm", "addr": "888 Broadway, New York, NY 10003", "country": "US"},
    # India
    {"name": "Flipkart Internet Pvt Ltd", "addr": "Buildings Alyssa, Begonia, Embassy Tech Village, Bangalore 560103", "country": "India"},
    {"name": "Zomato Media Private Limited", "addr": "Ground Floor, Tower C, Vipul Tech Square, Gurgaon 122002", "country": "India"},
    {"name": "Swiggy Bundl Technologies Ltd", "addr": "Tower D, IBC Knowledge Park, Bannerghatta Rd, Bangalore 560029", "country": "India"},
    {"name": "Singleton Test India Shop", "addr": "MG Road, Opposite Metro Pillar 120, Kochi 682016", "country": "India"},
    # France (Test Set includes France!)
    {"name": "LVMH Moet Hennessy Louis Vuitton SE", "addr": "22 Avenue Montaigne, 75008 Paris", "country": "France"},
    {"name": "TotalEnergies SE", "addr": "2 Place Jean Millier, La Defense 6, 92400 Courbevoie", "country": "France"},
    {"name": "BNP Paribas SA", "addr": "16 Boulevard des Italiens, 75009 Paris", "country": "France"},
    {"name": "Carrefour Hypermarches SAS", "addr": "93 Avenue de Paris, 91300 Massy", "country": "France"},
    {"name": "Societe Generale SA", "addr": "29 Boulevard Haussmann, 75009 Paris", "country": "France"},
    {"name": "Singleton Test France Boutique", "addr": "12 Rue de la Paix, 75002 Paris", "country": "France"},
]


def generate_variations(base_name: str, base_addr: str, count: int, country: str):
    """Generate realistic noisy record variations."""
    variations = []
    suffixes = {
        "US": ["Inc", "Corp", "LLC", "Co", "Company"],
        "India": ["Pvt Ltd", "Private Limited", "Ltd", "Enterprises"],
        "France": ["SA", "SARL", "SAS", "SE"],
    }.get(country, ["Inc", "Ltd"])

    for i in range(count):
        # Name noise: transposition or suffix change
        tokens = base_name.split()
        if len(tokens) >= 2 and i % 2 == 1:
            name_var = f"{tokens[1]} {tokens[0]} {' '.join(tokens[2:])}"
        else:
            name_var = f"{base_name} {random.choice(suffixes)}"

        # Address noise: abbreviation or missing address
        if i == 0 and random.random() < 0.25:
            addr_var = None  # Missing address
        else:
            addr_var = (
                base_addr.replace("Avenue", "Ave")
                .replace("Street", "St")
                .replace("Road", "Rd")
                .replace("Boulevard", "Blvd")
                .replace("Suite", "Ste")
            )

        variations.append((name_var, addr_var))
    return variations


def create_dataset_split(templates, s1_prefix, s2_prefix, s3_prefix, is_train=True):
    s1_rows = []
    s2_rows = []
    s3_rows = []
    gt_rows = []

    s2_counter = 1000
    s3_counter = 5000

    for idx, tmpl in enumerate(templates):
        s1_id = f"S1-{idx + 1:05d}"
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": tmpl["name"],
            "business_address": tmpl["addr"],
            "country": tmpl["country"],
        })

        if "Singleton" in tmpl["name"]:
            # Singletons have no matches
            if is_train:
                gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
            continue

        # Non-singletons match 1-2 in S2, 1-2 in S3
        num_s2 = random.choice([1, 2])
        num_s3 = random.choice([1, 2])

        s2_vars = generate_variations(tmpl["name"], tmpl["addr"], num_s2, tmpl["country"])
        s3_vars = generate_variations(tmpl["name"], tmpl["addr"], num_s3, tmpl["country"])

        matched_ids = []
        for name_v, addr_v in s2_vars:
            s2_counter += 1
            s2_id = f"S2-{s2_counter}"
            s2_rows.append({
                "entity_id": s2_id,
                "business_name": name_v,
                "business_address": addr_v if addr_v else "",
                "country": tmpl["country"],
            })
            matched_ids.append(s2_id)

        for name_v, addr_v in s3_vars:
            s3_counter += 1
            s3_id = f"S3-{s3_counter}"
            s3_rows.append({
                "entity_id": s3_id,
                "business_name": name_v,
                "business_address": addr_v if addr_v else "",
                "country": tmpl["country"],
            })
            matched_ids.append(s3_id)

        if is_train:
            gt_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched_ids),
            })

    return (
        pd.DataFrame(s1_rows),
        pd.DataFrame(s2_rows),
        pd.DataFrame(s3_rows),
        pd.DataFrame(gt_rows) if is_train else None,
    )


# Generate Train
df_s1_tr, df_s2_tr, df_s3_tr, df_gt_tr = create_dataset_split(
    TRAIN_TEMPLATES, "S1", "S2", "S3", is_train=True
)
df_s1_tr.to_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", index=False)
df_s2_tr.to_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", index=False)
df_s3_tr.to_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", index=False)
df_gt_tr.to_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", index=False)
print(f"Generated train: S1={len(df_s1_tr)}, S2={len(df_s2_tr)}, S3={len(df_s3_tr)}, GT={len(df_gt_tr)}")

# Generate Test
df_s1_te, df_s2_te, df_s3_te, _ = create_dataset_split(
    TEST_TEMPLATES, "S1", "S2", "S3", is_train=False
)
df_s1_te.to_csv(TEST_DIR / "test_source1.tsv", sep="\t", index=False)
df_s2_te.to_csv(TEST_DIR / "test_source2.tsv", sep="\t", index=False)
df_s3_te.to_csv(TEST_DIR / "test_source3.tsv", sep="\t", index=False)
print(f"Generated test: S1={len(df_s1_te)}, S2={len(df_s2_te)}, S3={len(df_s3_te)}")
