#
# WHAT THIS SCRIPT DOES
# ---------------------
# Takes the Drugs.com side-effects dataset (one row per drug, with all side
# effects crammed into a single 'side_effects' string) and produces a clean,
# "long" table with one row per (drug, side effect) pair, where near-duplicate
# side-effect terms have been merged onto a single canonical term.
#
# THE PIPELINE, STEP BY STEP
# --------------------------
#   1. LOAD + PREPROCESS
#      Read the CSV, drop junk columns, fix numeric columns, and collapse
#      formulation-variant drug names.
#
#   2. TEXT CLEANING
#      Split each drug's side-effect string into individual terms, then scrub
#      each one: strip lead-in boilerplate ("get emergency medical help..."),
#      remove web-scraping artifacts (JavaScript, CSS), normalise wording
#      ("your" -> "", "a" -> ""), and drop anything not shaped like a real
#      side effect (too short, too long, or matching an ingredient/junk list).
#      Result: `long_se`, one (drug, side_effect) row per pair, deduplicated.
#
#   3. CLUSTERING, PASS 1
#      Vectorise unique terms with TF-IDF and cluster with DBSCAN (cosine
#      distance) so near-identical terms ("trouble breathing" vs
#      "difficulty breathing") land together. Manually reviewed "bad" clusters
#      (semantic false positives) are excluded, and the smallest/cleanest term
#      in each cluster is chosen as the canonical representative.
#
#   4. CLUSTERING, PASS 2 AND PASS 3
#      Re-run DBSCAN on the remaining terms after removing bad clusters, to
#      catch merges that only became visible once noisy terms were gone.
#      (Each pass shrinks the vocabulary; the loop ends when the operator is
#      happy with the cluster review output.)
#
#   5. FINAL MERGE
#      `merge_after_second_pass` builds a lookup table (term -> canonical),
#      applies operator decisions (clusters to skip, winners to override),
#      remaps `final`, and deduplicates.
#
#   6. SYNONYM MAP
#      A last, hand-maintained pass applied to the finished table. Anything
#      the automated clustering couldn't merge correctly is fixed here.
#
# WHY THE HUMAN-IN-THE-LOOP BITS EXIST
# ------------------------------------
# DBSCAN on TF-IDF is a fuzzy tool: it merges "muscle pain" and "muscle
# weakness" just as happily as "tremor" and "tremors". The bad_clusters lists,
# the winner_overrides dict, and the synonym map are all manual review
# mechanisms: the cluster prints let a human inspect each merge group and
# veto or redirect it.

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# =============================================================================

# Load CSV and remove unnamed columns
df = pd.read_csv("drugs_side_effects_drugs_com.csv", index_col=False)
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Remove non-essential columns
df = df.drop(columns=["drug_link", "medical_condition_url"])

# Convert numeric columns, handle comma separators, and coercion errors
df["no_of_reviews"] = pd.to_numeric(
    df["no_of_reviews"].astype(str).str.replace(",", "").str.strip(),
    errors="coerce"
)
df["rating"] = pd.to_numeric(
    df["rating"].astype(str).str.replace(",", "").str.strip(),
    errors="coerce"
)

# Collapse variations of the same drug into one
DRUG_SYNONYM_MAP = {
    "zyprexa zydis": "zyprexa",
}

df["drug_name"] = (
    df["drug_name"]
    .str.strip().str.lower()
    .replace(DRUG_SYNONYM_MAP)
)

# =============================================================================
# SECTION 2: TEXT CLEANING PIPELINE DEFINITIONS
# =============================================================================

# Length constraints: filters to remove fragmented or overly verbose side effects
LENGTH_CONSTRAINTS = {
    "max_effect_length": 45,
    "min_effect_length": 6,
    "max_words_per_effect": 5,
}

# Phrases to remove before any splitting occurs to catch leftover HTML/JS code
PRE_SPLIT_PATTERNS = [
    r'https?://\S+',
    r'\bwww\.[a-z0-9.-]+\.[a-z]{2,}\S*',
    r'\bfunction\b[^;}{]*',
    r'\bvar\s+[^;]*;',
    r'@?[a-z_$][\w$]*\.(?:push|append|appendChild|'
    r'removeChild|getElementById|querySelector|'
    r'addEventListener|setAttribute|createElement)'
    r'\s*\([^;]{0,200}',
    r'\b(?:window|document)\.\w+',
    r'\bdataLayer\b|\bgoogle_tag|\bgpt\b',
    r'\bconsole\.\w+\s*\([^;]{0,200}',
    r'\b(?:get|set)Time(?:out)?\b',
    r'\b(?:if|else|return|try|catch|new)\b(?=\s*[\(\w])',
]

# Strips lead-ins that prefix many side effect strings
LEAD_IN_RE = re.compile(
    r"^(get emergency medical help|seek medical (treatment|help|attention)"
    r"|call your doctor[\w\s,]*"
    r"|check with your (doctor|health ?care professional|healthcare professional)[\w\s,]*"
    r"|stop using [\w\s]+|common side effects of [\w\s]+"
    r"|[\w\s]*side effects may include|symptoms may include"
    r"|[a-z\s]+ may cause (serious )?side effects"
    r"|this reaction may occur[\w\s]*"
    r"|some side effects may occur[\w\s]*"
    r"|other side effects not listed[\w\s]*"
    r"|you may report side effects[\w\s]*"
    r"|if you notice any other effects[\w\s,.]*"
    r")[:.\s]*"
    r"|along with its needed effects[\w\s,]*"
    r"|(a )?medicine may cause some unwanted effects[\w\s,.]*",
    re.I
)

# Frequency section headers embedded in the text, replaced with period
SECTION_HEADERS = re.compile(
    r"\b(less common( or rare)?|more common|incidence not known"
    r"|symptoms of (overdose|resorcinol poisoning)[\w\s]*"
    r")\b[:,]?\s*",
    re.I
)

# Common endings of side effect strings
TRAILING_TITLE_RE = re.compile(r"\b[a-z]+\s+side effects\s*$", re.I)

# Drop the allergic reaction boilerplate
ALLERGIC_SYMPTOMS = re.compile(
    r"^.*signs of an allergic reaction[,:]?\s*", re.I
)

# Patterns recognising scraper leftovers from CSS, HTML, URLs, etc.
CODE_CONFIG_PATTERNS = [
    r'[{}<>]',
    r'["\']',
    r'[/:_]',
    r'&&|\|\||!=?|=+',
    r'\b(function\s*\(|\.prototype\b|console\.)\b',
    r'\b(getElement|querySelector|setAttribute|addEventListener)\b',
    r'\b(var\s+\w+\s*=|=\s*"\w+)\b',
    r'px$|\d+px',
    r'\b(?=[a-z0-9]*\d)[a-z0-9]{6,}\b',
    r'\batob\(',
    r'\b(if|else|return|try|catch)\b',
    r'\bhttps?:',
    r"(\.prototype\b|atob\(|btoa\(|append\s*child|remove\s*child"
    r"|querySelector|getElementById|addEventListener|\.push\("
    r"|document\.|window\.|function\s*\(|ddc-reference|reference\.?\s*list"
    r"|\.append\(|\.remove\(|\{\s*[\w-]+\s*:"
    r"|console\.|=>|\$\(|===|!==)"
]

# Ingredients and dosages that aren't side effects
INGREDIENT_PATTERNS = [
    r'\b(sodium|chloride|lactose)\b',
    r'\b(mg|g|ml|%|tablet|capsule|injection|antibiotic[s]?)\b',
    r'\b(syrup|solution|gel|cream|ointment|advil|dose[s]?|vapo)\b',
]

# Other classic boilerplate phrases that were found
BOILERPLATE_PATTERNS = [
    r'\b(information|storage|protect|children|vitamin[s]?|prescription[s]?)\b',
    r'\b(devices|available|flavoring|smith|kline|reserved|share|html|mailto|lazy)\b',
    r'\b(alert[s]?|newsletter[s]?|update[s]?)\b',
    r'\b(healthcare|professional[s]?|pharmacist[s]?)\b',
    r'\b(do not|don\'t).*freeze\b',
    r'\bprotect.*from.*(light|moisture|heat)\b',
    r'\b(care provider|emergency medical|medical help|medical attention)\b',
    r'\b(dose|months after|weeks after|last dose)\b',
    r'\b(needed effects|unwanted effects|medicine may cause)\b',
    r'\balong with its needed effects\b',
    r'\bstop using this medicine|stop using\b',
    r'\bmay cause serious side effects\b',
    r'\bcommon side effects of\b',
    r'\bsome side effects may occur during\b',
    r'\bany signs of a new infection\b',
    r'\bwhich can be fatal\b',
    r'\bstop taking this medicine\b',
    r'\bhad no reaction\b',
    r'\buse caution to avoid falling\b',
    r'\bwill not cause [any]? harm\b',
    r'\bthis may include [walking]?\b',
    r'\byou may have none at all\b',
    r'\bdeath may occur\b'
]

# Short keywords that are often left behind by longer junk phrases
JUNK_KEYWORDS = [
    "bothersome", "count",
    "modal", "group", "inset", "media", "table", "query", "lower",
    "multi", "propa", "kwik", "log", "nitro", "covid",
    "chlor", "nytol"
]

# Genuine one-word side effects that would otherwise fail minimum length checks
VALID_SHORT_WORDS = {
    "rash", "pain", "itch", "fever", "cough", "hives",
    "swelling", "bruising", "blistering", "peeling", "numbness",
    "vomiting", "diarrhea", "dizziness", "nausea", "vertigo",
    "insomnia", "anorexia", "itching", "bleeding", "bloating"
}

# Gets rid of fragments that are just the drug name
DRUG_TOKENS = {
    tok
    for name in df["drug_name"].dropna().unique()
    for tok in re.findall(r"[a-z]{5,}", name)
} - VALID_SHORT_WORDS

def is_drug_name_leak(effect: str) -> bool:
    """Term opens with a known drug name -> it's a sentence fragment."""
    if not effect:
        return True
    return effect.split()[0] in DRUG_TOKENS

# Terms that passed the filters but are still unwanted
EXPLICIT_BLACKLIST = {
    "along with its needed effects", "unwanted effects", "needed effects",
    "people who", "in rare cases", "rare cases", "sometimes death",
    "does not go away", "especially on face", "in those who are debilitated",
    "green in color", "in some cases", "aqua gem", "at night", "not at all",
    "keep all drugs in safe place", "mark clear", "in back", "more often",
    "store in dry place", "around eyes", "reduce some of these side effects",
    "without fever", "report any new", "accidental injury", "worsening symptoms to doctor",
    "like feeling", "allergic reaction can be severe", "stop using unisom sleep gels",
    "ketamine may cause serious side effects", "in premature", "nevi you already have",
    "interferon gamma", "often with a purple center red", "changes in shape",
    "call doctor for medical advice", "peeling skin or without fever"
}

# Bare body parts that pass filters but are not side effects
BODY_PARTS = {
        "eye", "eyes", "hand", "hands", "foot", "feet", "leg", "legs",
        "arm", "arms", "finger", "fingers", "toe", "toes", "lip", "lips",
        "tongue", "mouth", "nose", "shoulder", "shoulders", "chest",
        "back", "stomach", "head", "neck", "ear", "ears", "thigh", "thighs",
        "knee", "knees", "wrist", "wrists", "ankle", "ankles", "hip", "hips",
        "waist", "belly", "breast", "breasts", "face", "bone", "throat",
        "body", "fingernail beds", "fingernail", "fingernails"
    }

# Regex helpers
SPACE_RE = re.compile(r"\s+")

EDGE_PUNCT_RE = re.compile(
    r"^[^\w\s]+|[^\w\s]+$"
)

# Strip leading filler words like conjunctions, articles, etc.
LEADING_WORDS_RE = re.compile(
    r"^(?:"
    r"or|and|with|including|such|a|an|the|your|that|is|are|"
    r"as|signs?|"
    r"ongoing|sudden|new|any|"
    r"mild|moderate|severe|serious|unusual|persistent|"
    r"very bad"
    r")\s+",
    re.I
)

# Remove any 'your' or 'the' wherever they appear to make a common form
YOUR_THE_RE = re.compile(
    r"\b(your|the)\b",
    flags=re.I
)

# Remove residual lead-in clauses
MAY_INCLUDE_RE = re.compile(
    r"^(at once|if you have|may include|that can affect)[\w\s:]*",
    flags=re.I
)

# Collapse repeated words that appeared after text scraping
DUPLICATE_WORD_RE = re.compile(
    r"\b(\w+)(\s+\1\b)+",
    flags=re.I
)

# Strip any remaining leading articles
ARTICLE_RE = re.compile(
    r"^(?:a|an|the)\s+",
    flags=re.I
)

# CamelCase terms scraped without space
LOWER_UPPER_RE = re.compile(
    r"([a-z])([A-Z])"
)

# Sepearators found: periods, semicolons, commas, colons, '-', and '--'
SEPARATOR_RE = r"[.;,:]|\n|\s-\s|-{2,}|•"
PRE_SPLIT_REGEX = re.compile("|".join(PRE_SPLIT_PATTERNS), re.I)
INGREDIENT_REGEX = re.compile("|".join(INGREDIENT_PATTERNS), re.I)
BOILERPLATE_REGEX = re.compile("|".join(BOILERPLATE_PATTERNS), re.I)
JUNK_REGEX = re.compile(r"\b(?:" + "|".join(JUNK_KEYWORDS) + r")\b", re.I)
CODE_REGEX = re.compile("|".join(CODE_CONFIG_PATTERNS), re.I)
ALPHANUMERIC_MIX_PATTERN = re.compile(
    r'^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]+$', re.I
)

def strip_code_segments(text: str) -> str:
    if not isinstance(text, str):
        return text
    return PRE_SPLIT_REGEX.sub(" ", text)

def clean_cell(text: str) -> str:
    """First stage cleaning, applied to each raw term right after splitting.
    Order matters: allergic reaction boilerplate goes before anything else and the
    CamelCase fix runs before split-term text is used."""
    txt = str(text)

    txt = ALLERGIC_SYMPTOMS.sub("", txt)
    txt = SPACE_RE.sub(" ", txt).strip()
    txt = LEADING_WORDS_RE.sub("", txt)
    txt = SECTION_HEADERS.sub(". ", txt)
    txt = TRAILING_TITLE_RE.sub("", txt)

    txt = LOWER_UPPER_RE.sub(r"\1. \2", txt)

    txt = txt.replace("(", " ")
    txt = txt.replace(")", " ")

    return txt

def is_junk(effect: str) -> bool:
    """Second-stage filter: is it a real side effect?
    The checks are ordered by cost and confidence: exact-match lists first, then regex groups,
    then structural rules.
    Returns True = drop the term"""
    if not effect or not isinstance(effect, str):
        return True
    effect_clean = effect.lower().strip()

    if effect_clean in EXPLICIT_BLACKLIST:
        return True
    if effect_clean in VALID_SHORT_WORDS:
        return False
    if effect_clean in BODY_PARTS:
        return True
    if INGREDIENT_REGEX.search(effect_clean):
        return True
    if BOILERPLATE_REGEX.search(effect_clean):
        return True
    if JUNK_REGEX.search(effect_clean):
        return True
    if ALPHANUMERIC_MIX_PATTERN.search(effect_clean):
        return True
    if re.search(r'\b\d+\b', effect_clean):
        return True
    if "°" in effect_clean:
        return True
    length = len(effect_clean)
    if length < LENGTH_CONSTRAINTS["min_effect_length"]:
        return True
    if length > LENGTH_CONSTRAINTS["max_effect_length"]:
        return True
    if len(effect_clean.split()) > LENGTH_CONSTRAINTS["max_words_per_effect"]:
        return True

    return False

def is_code_artifact(text: str) -> bool:
    """Catch scraper/code leftovers that survived the junk filters.
    Separate from is_junk because these checks apply to the term as is,
    e.g. lowercase 6+ char alphanumeric blobs."""
    if not text or not isinstance(text, str):
        return True
    if CODE_REGEX.search(text.lower()):
        return True
    if re.match(r'^[a-z0-9]{6,}$', text):
        return True
    if len(text) <= 3 and re.match(r'^[a-z0-9]+$', text):
        return True

    return False

def normalise_side_effect_text(text: str) -> str:
    """Normalise a cleaned term to its canonical surface form.
    Lowercase, strip punctuation, remove filler words, collapse repeated words."""

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = EDGE_PUNCT_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip()
    text = YOUR_THE_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip()
    text = MAY_INCLUDE_RE.sub("", text).strip()
    text = DUPLICATE_WORD_RE.sub(r"\1", text)
    text = ARTICLE_RE.sub("", text)
    text = LEADING_WORDS_RE.sub("", text)

    return text

def extract_side_effects(df_raw, split_pattern=SEPARATOR_RE):
    """Explode the raw dataframe into one row per side effect.
    Pipeline per term: split -> clean_cell -> normalise -> keep only
    terms that pass is_junk AND is_code_artifact -> deduplicate drug, side effect pairs."""
    df = df_raw.copy()
    df = df.assign(
        side_effects=df["side_effects"].str.split(split_pattern, regex=True)
    ).explode("side_effects").reset_index(drop=True)

    df["side_effects"] = df["side_effects"].apply(clean_cell)
    df["side_effects"] = df["side_effects"].apply(normalise_side_effect_text)

    mask = df["side_effects"].apply(
        lambda x: not is_junk(x) and not is_code_artifact(x) and not is_drug_name_leak(x)
    )
    return (df[mask]
            .drop_duplicates(subset=["drug_name", "side_effects"])
            .reset_index(drop=True))

# Run initial cleaning stage
long_se = extract_side_effects(df)
print("PRE-FILTERING:")
print(f'Unique drugs represented: {df["drug_name"].nunique()}')
print(f'Unique medical conditions treated: {df["medical_condition"].nunique()}')

print("\nPOST-FILTERING:")
print(f"Total side effects represented: {len(long_se)}")
print(f'Unique drugs represented: {long_se["drug_name"].nunique()}')
print(f'Unique medical conditions treated: {long_se["medical_condition"].nunique()}')
print(f"Unique side effects: {long_se['side_effects'].nunique()}")

print(f"\nTop 10 most common side effects:")
top_side_effects = long_se["side_effects"].value_counts().head(10)
print(top_side_effects)

print(long_se[['drug_name', 'side_effects']].head(50).to_string())

# =============================================================================
# SECTION 3: FIRST CLUSTERING PASS
# =============================================================================
# Vectorise unique terms and cluster with DBSCAN using cosine distance.
# eps=0.3 means terms within ~72 degrees of each other join the same cluster
# min_samples=2 kills single-term noise
# Cluster -1 = noise (no close neighbours found)
terms = long_se["side_effects"].drop_duplicates()
print(len(terms))

X = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
).fit_transform(terms)

clusters = DBSCAN(
    metric="cosine",
    eps=0.3,
    min_samples=2
).fit_predict(X)

cluster_df = pd.DataFrame({
    "term": terms,
    "cluster": clusters
})

# Print every non-noise cluster to be reviewed what DBSCAN merges
print(f"Number of clusters: {cluster_df['cluster'].nunique()}")

for c in sorted(cluster_df["cluster"].unique()):
    if c == -1:
        continue

    print(f"\nCLUSTER {c}")

    cluster_terms = cluster_df.loc[
        cluster_df["cluster"] == c,
        "term"
    ]

    print(cluster_terms.tolist())

cluster_sizes = (
    cluster_df["cluster"]
    .value_counts()
    .sort_values(ascending=True)
)

print(cluster_sizes.head(30))

# Print the smallest clusters first - most likely to be genuine near-duplicate pairs that can be merged
for c in cluster_sizes.index[:20]:
    if c == -1:
        continue

    print(f"\nCLUSTER {c}")
    print(cluster_df.loc[
        cluster_df["cluster"] == c,
        "term"
    ].tolist())

# Choosing a canonical winner per cluster
# Rule: the term with the fewest words (then fewest chars, then alphabetical) wins
cluster_df["n_words"] = cluster_df["term"].str.split().str.len()
cluster_df["n_chars"] = cluster_df["term"].str.len()

size = cluster_df.groupby("cluster")["term"].transform("size")
pair = cluster_df["cluster"].ne(-1) & size.eq(2)

ordered = (cluster_df.loc[pair]
           .sort_values(["cluster", "n_words", "n_chars", "term"], kind="mergesort"))

winner = ordered.groupby("cluster", as_index=False).head(1)
canon_by_cluster = dict(zip(winner["cluster"], winner["term"]))

cluster_df["canonical"] = [
    canon_by_cluster.get(c, t) for c, t in zip(cluster_df["cluster"], cluster_df["term"])
]

# LUT: every term -> its cluster winner (term maps to itself if not in cluster)
# This is what gets applied to the long dataframe
lut = dict(zip(cluster_df["term"], cluster_df["canonical"]))

missing = ~long_se["side_effects"].isin(lut)
print("unmatched terms:", missing.sum())

# Apply the pass-1 merge and collapse the now-duplicate rows
long_se["side_effects"] = long_se["side_effects"].map(lambda t: lut.get(t, t))

before = len(long_se)
long_se = long_se.drop_duplicates(subset=["drug_name", "side_effects"]).reset_index(drop=True)
print(f"{before} -> {len(long_se)} rows ({before - len(long_se)} collapsed)")

# Diagnostics on ties
# Clusters whose two terms have the same word count: the char-count tiebreak picked the winner (weaker signal)
g = cluster_df.loc[pair].groupby("cluster")["n_words"]
tied = g.nunique().loc[lambda s: s.eq(1)].index

out = (cluster_df.loc[pair & cluster_df["cluster"].isin(tied)]
       .sort_values(["cluster", "n_words", "n_chars", "term"])
       [["cluster", "term", "n_words"]]
       .reset_index(drop=True))

print(out.to_string(max_rows=None))

n_pairs = pair.sum() // 2
print(f"{n_pairs} size-2 clusters, {len(tied)} tied on word count "
      f"({len(tied)/n_pairs:.0%})")

# Manual review of first pass
# Clusters inspected above that are semantic false positives, left unmerged downstream
bad_clusters = [65, 95, 194, 239, 243, 246, 247, 248, 249, 256, 297]

cluster_terms = cluster_df.loc[cluster_df["cluster"].isin(bad_clusters), "term"]

long_se_1 = long_se[~long_se["side_effects"].isin(cluster_terms)].copy()
terms_1 = long_se_1["side_effects"].drop_duplicates()
print(len(terms_1))

# =============================================================================
# SECTION 4a: SECOND CLUSTERING PASS
# =============================================================================
# Same procedure on the reduced vocabulary
# Removing bad-cluster terms changes the TF-IDF space so potential for new clusters to emerge
X = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
).fit_transform(terms_1)

clusters_1 = DBSCAN(
    metric="cosine",
    eps=0.3,
    min_samples=2
).fit_predict(X)

cluster_df = pd.DataFrame({
    "term": terms_1,
    "cluster": clusters_1
})

print(f"Number of clusters: {cluster_df['cluster'].nunique()}")

for c in sorted(cluster_df["cluster"].unique()):
    if c == -1:
        continue

    print(f"\nCLUSTER {c}")

    cluster_terms = cluster_df.loc[
        cluster_df["cluster"] == c,
        "term"
    ]

    print(cluster_terms.tolist())

# Manual review of the second pass
bad_clusters = [0, 35, 44, 52, 70, 77, 93, 96, 105, 107, 108, 122]

cluster_terms = cluster_df.loc[cluster_df["cluster"].isin(bad_clusters), "term"]

long_se_2 = long_se_1[~long_se_1["side_effects"].isin(cluster_terms)].copy()
terms_2 = long_se_2["side_effects"].drop_duplicates()
print(len(terms_2))

# =============================================================================
# SECTION 4b: THIRD CLUSTERING PASS
# =============================================================================
# Third and final iteration
X = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
).fit_transform(terms_2)

clusters_2 = DBSCAN(
    metric="cosine",
    eps=0.3,
    min_samples=2
).fit_predict(X)

cluster_df = pd.DataFrame({
    "term": terms_2,
    "cluster": clusters_2
})

print(f"Number of clusters: {cluster_df['cluster'].nunique()}")

# =============================================================================
# SECTION 5: FINAL CLUSTER MERGE
# =============================================================================

# Prints every mergeable cluster with its members and the term the automatic rule would pick as winner.
# This is the human decision list: read it, then encode decisions as bad_clusters (don't merge) or
# winner_overrides (merge but pick a different representative)
review = cluster_df[cluster_df["cluster"].ne(-1)].copy()
size = review.groupby("cluster")["term"].transform("size")
review = review[size.ge(2)]

def winner_of(c):
    grp = (review[review["cluster"] == c]
           .assign(n_words=lambda d: d["term"].str.split().str.len(),
                   n_chars=lambda d: d["term"].str.len())
           .sort_values(["n_words", "n_chars", "term"], kind="mergesort"))
    return grp.iloc[0]["term"]

for c, grp in review.groupby("cluster"):
    terms = grp["term"].tolist()
    print(f"\nCLUSTER {c} (size {len(terms)}) -> winner: {winner_of(c)!r}")
    for t in sorted(terms):
        marker = "  <-- winner" if t == winner_of(c) else ""
        print(f"    {t!r}{marker}")

def merge_after_second_pass(
    long_df,
    cluster_df,
    bad_clusters=(),
    min_cluster_size=2,
    max_cluster_size=None,
    winner_overrides=None,
    verbose=True,
):
    """Apply the final clustering decisions to the long dataframe.
    Parameters:
    ----------
    long_df : dataframe with 'drug_name' and 'side_effects' columns
    cluster_df : 'term' + 'cluster' assignments from the final DBSCAN pass
    bad_clusters : cluster IDs to leave unmerged (false positives)
    min_cluster_size : clusters smaller than this are left unmerged
    max_cluster_size : optional cap as suspiciously large clusters are usually chains of loosely related terms
    winner_overrides : {cluster_id : term} - force a specific representative for a cluster whose automatic winner is th wrong choice

    Returns (merged_df, lut) where lut is the full term -> canonical mapping, useful for auditing exactly which terms are renamed."""
    cdf = cluster_df.copy()
    winner_overrides = winner_overrides or {}

    # Keep single word clusters unchanged
    size = cdf.groupby("cluster")["term"].transform("size")
    unreliable = (
        cdf["cluster"].eq(-1)
        | cdf["cluster"].isin(bad_clusters)
        | size.lt(min_cluster_size)
    )
    if max_cluster_size is not None:
        unreliable |= size.gt(max_cluster_size)

    cdf.loc[unreliable, "cluster"] = -1

    cdf["n_words"] = cdf["term"].str.split().str.len()
    cdf["n_chars"] = cdf["term"].str.len()

    ordered = (cdf[cdf["cluster"].ne(-1)]
               .sort_values(["cluster", "n_words", "n_chars", "term"],
                            kind="mergesort"))
    winners = ordered.groupby("cluster", as_index=False).head(1)
    for c, term in winner_overrides.items():
        if c in winners["cluster"].values:
            winners.loc[winners["cluster"] == c, "term"] = term
        else:
            print(f"warning: no cluster {c} to override")
    canon_by_cluster = dict(zip(winners["cluster"], winners["term"]))

    lut = {
        t: canon_by_cluster.get(c, t)
        for t, c in zip(cdf["term"], cdf["cluster"])
    }

    def resolve(t, _seen=None):
        seen = _seen or set()
        while t in lut and t not in seen and lut[t] != t:
            seen.add(t)
            t = lut[t]
        return t

    lut = {t: resolve(c) for t, c in lut.items()}

    out = long_df.copy()
    out["side_effects"] = out["side_effects"].map(lambda t: lut.get(t, t))

    before = len(out)
    out = (out.drop_duplicates(subset=["drug_name", "side_effects"])
              .reset_index(drop=True))

    if verbose:
        n_merged = sum(1 for t, c in lut.items() if t != c)
        print(f"terms remapped: {n_merged}")
        print(f"rows: {before} -> {len(out)} ({before - len(out)} collapsed)")
        print(f"unique side effects: {out['side_effects'].nunique()}")
        if bad_clusters:
            print(f"bad clusters left unmerged: {sorted(bad_clusters)}")

    return out, lut

# =============================================================================
# SECTION 6: SYNONYM MAPPING
# =============================================================================
# Synonym map applied after cluster merging as the very last transformation
SYNONYM_MAP = {
    "like you might pass out" : "feeling like you might pass out",
    "trouble breathing": "difficulty breathing",
    "shortness of breath": "difficulty breathing",
    "breathing difficulty": "difficulty breathing",
    "difficult breathing": "difficulty breathing",
    "short of breath": "difficulty breathing",
    "yellowing of skin": "jaundice",
    "flu symptoms" : "flu-like symptoms",
    "cold symptoms" : "flu-like symtpoms",
    "upper stomach" : "stomach pain",
    "tired feeling" : "feeling tired",
    "feeling sleepy" : "feeling tired",
    "sores in mouth" : "mouth sores",
    "of skin infection" : "",
    "feeling nervous" : "feeling anxious",
    "fast or slow" : "fast or slow heartbeats",
    "fast or slow heartbeat" : "fast or slow heartbeats",
    "feet sore throat sores" : "throat pain",
    "diarrhea may be bloody" : "bloody diarrhea",
    "light-headed" : "light-headed feeling",
    "like rash" : "rash",
    "seizure convulsions" : "convulsions",
    "sleep problems insomnia" : "insomnia",
    "feeling light-headed" : "light-headed feeling",
    'high potassium slow heart rate' : "slow heart rate",
    "tongue red" : "red tongue"
    }

# Run the final merge
final, final_lut = merge_after_second_pass(
    long_se_2,
    cluster_df,
    bad_clusters=[1, 7, 9, 18, 20, 41],
    winner_overrides={
    24 : "black or tarry stools",
    32 : "urination problems",
    33 : "pain in stomach and back",
    37 : "abnormal vision",
    39 : "upper stomach pain",
    47 : "sores in mouth",
    57 : "blue-colored skin",
    93 : "involuntary eye movements",
    115 : "indigestion or stomach pain"}
)

swaps = pd.Series(final_lut)
swaps = swaps[swaps.index != swaps]

clusters_to_delete = [107]
delete_terms = set(
    cluster_df.loc[cluster_df["cluster"].isin(clusters_to_delete), "term"]
)

final = final[~final["side_effects"].isin(delete_terms)].copy()

# Apply the synonym map then deduplicate again
final["side_effects"] = final["side_effects"].replace(SYNONYM_MAP)

before = len(final)
final = (final.drop_duplicates(subset=["drug_name", "side_effects"])
              .reset_index(drop=True))
print(f"{before} -> {len(final)} rows ({before - len(final)} collapsed by synonyms)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
# Compare dataset shape at each stage to sanity-check the pipeline: the raw
# drug/condition counts should stay stable, while unique side-effect terms
# shrink as cleaning and merging collapse variants onto canonical forms.
print("RAW DF:")
print(f'Unique drugs represented: {df["drug_name"].nunique()}')
print(f'Unique medical conditions treated: {df["medical_condition"].nunique()}')

print("\nPOST-TEXT CLEANING:")
print(f"Total side effects represented: {len(long_se)}")
print(f'Unique drugs represented: {long_se["drug_name"].nunique()}')
print(f'Unique medical conditions treated: {long_se["medical_condition"].nunique()}')
print(f"Unique side effects: {long_se['side_effects'].nunique()}")
print(f"\nTop 10 most common side effects:")
top_side_effects_old = long_se["side_effects"].value_counts().head(10)
print(top_side_effects_old)

print("\nPOST-CLUSTERING:")
print(f"Total side effects represented: {len(final)}")
print(f'Unique drugs represented: {final["drug_name"].nunique()}')
print(f'Unique medical conditions treated: {final["medical_condition"].nunique()}')
print(f"Unique side effects: {final['side_effects'].nunique()}")

print(f"\nTop 10 most common side effects:")
top_side_effects = final["side_effects"].value_counts().head(10)
print(top_side_effects)

final.to_csv("long_side_effects.csv", index=False)