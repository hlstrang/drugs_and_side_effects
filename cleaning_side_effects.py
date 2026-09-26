import pandas as pd
import re

BOILERPLATE_PHRASES = [
    "along with its needed effects",
    "see important information",
    "important information",
    "complete list of side effects",
    "call your doctor",
    "get medical help",
    "seek emergency",
    "read more",
    "medical news today",
    "this document does not contain",
    "tell your doctor",
    "report side effects",
    "unwanted side effects",
]

PROTECTED = {
    "little or no urination": "urination decreased",
    "watery or bloody": "watery stool",
    "bloody or tarry": "blood in stool",
    "loss of appetite": "appetite loss",
    "ringing in your ears": "tinnitus",
}

BOILER_TOKEN_BLACKLIST = {
    "along with its needed effects", "unwanted effects", "needed effects",
    "including: see important information above", "important information",
    "see important information above", "see full prescribing information",
    "important safety information", "call your doctor", "common side effects of"
    "please see", "read more", "click here", "see warning", "reduce some of these side effects",
    "people who", "time use", "you could develop vitamin b", "green in color",
    "wash skin", "of liver", "changes in shape", "sometimes death", "use caution to avoid feeling",
    "having sex", "alcohol", "accidental injury", "very weak", "within day", "child's height",
    "will not cause any harm", "in those who are debilitated", "especially if you are woman",
    "older adult", "high doses", "like other narcotic medications", "rectal area",
    "especially when used on face", "longer afterward", "use caution to avoid falling",
    "making phone calls", "breakthrough bleeding", "people who are debilitated",
    "aggressive as: mood", "other irritation", "common side effects of acetaminophen",
    "which can be fatal", "as: mood", "if you feel agitated", "often with purple center red",
    "main blood artery of body", "around eyes", "around mouth", "in rare cases", "rare cases",
    "report any new", "does not go away", "especially on face", "common side effects of hydrochlorothiazide",
    "those who are overweight", "of fluid", "later had no memory of it"
}

def strip_boilerplate(txt):
    txt = re.sub(r"([a-z])([A-Z])", r"\1. \2", str(txt))
    sentences = re.split(r"(?<=[.!])\s+", str(txt))
    kept = []
    for s in sentences:
        sl = s.lower()
        allergy_templ = ("hives" in sl and "breathing" in sl)
        resp_templ = any(k in sl for k in [
            "unusual hoarseness",
            "tightness in the chest or throat",
            "signs of an allergic reaction",
        ])
        boiler = any(p in sl for p in BOILERPLATE_PHRASES)
        if not (allergy_templ or resp_templ or boiler):
            kept.append(s)
    return " ".join(kept)

def mask_phrases(txt):
    for i, phrase in enumerate(PROTECTED.keys()):
        txt = re.sub(re.escape(phrase), f"__PM{i}__", str(txt), flags=re.I)
    return txt

def is_junk(effect):
    return (
        effect in BOILER_TOKEN_BLACKLIST
        or len(effect) > 45
        or any(h in effect for h in ["see ", "information", "attention",
                                     "doctor", "medicin", "reaction:", "also", "signs",
                                     "black"])
    )

raw = pd.read_csv("cleaned_df.csv")
raw = raw.drop_duplicates(subset=["drug_name"]).reset_index(drop=True)

raw["side_effects_clean"] = raw["side_effects"].apply(strip_boilerplate)

raw["side_effects_clean"] = raw["side_effects_clean"].apply(mask_phrases)

for i, canon in enumerate(PROTECTED.values()):
    raw["side_effects_clean"] = raw["side_effects_clean"].str.replace(
        f"__PM{i}__", canon, case=False, regex=False
    )

df = raw.assign(
    side_effects=raw["side_effects_clean"].str.split(r"[-,;.\n]\s*|\s+(?:or|and)\s+")
).explode("side_effects")

df["side_effects"] = (
    df["side_effects"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"^[^\w\s]+|[^\w\s]+$", "", regex=True)
)

df["side_effects"] = (
    df["side_effects"]
    .str.replace(r"^(get|call|seek|check|signs|of|your|a|an|that|is|are|feeling|having)\s+", "", regex=True)
    .str.replace(r"\b(your|the|a|an)\b", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df[df["side_effects"].str.len() > 0]
df = df[~df["side_effects"].str.contains(r"\b(?:may|occur|contact|this|reaction|help|using|medicine|notice|professional|healthcare)\b", na=False)]
df = df[df["side_effects"].str.split().str.len() <= 6]
df["side_effects"] = df["side_effects"].str.replace(
    r"^(or|and|with|including|such)\s+", "", regex=True
)

synonym_map = {
    # Breathing
    "trouble breathing": "difficulty breathing",
    "shortness of breath": "difficulty breathing",
    "breathing difficulty": "difficulty breathing",
    "difficult breathing": "difficulty breathing",
    "short of breath": "difficulty breathing",

    # Skin & Jaundice
    "yellowing of skin": "jaundice",
    "yellowing skin": "jaundice",
    "swelling of face": "facial swelling",

    # Plurals / Variations
    "headaches": "headache",
    "muscle aches": "muscle pain",
    "muscle ache": "muscle pain",
    "diarrhea that is watery stool": "watery stool",
    "12 deficiency" : "B12 deficiency",
    "pins" : "pins and needles",
    "if you feel impulsive" : "impulsiveness",
    "of liver problems like dark urine" : "dark urine",
    "leading to kidney failure" : "kidney failure",
    "codeine can slow breathing" : "slow breathing",
    "vagina): burning" : "vaginal burning",
    "agitated" : "agitation",
    "cloudy urine fever with" : "cloudy urine",
    "as: agitation" : "agitation",
    'pruned" appearance of skin' : "pruned appearance of skin",
    "flaking" : "skin flaking",
    "anxious" : "anxiety",
    "any other changes in hearing" : "changes in hearing",
    "cold symptoms such as stuffy nose" : "stuffy nose",
    "cold symptoms such as runny" : "runny nose",
    "excited (especially in children" : "excitable",
    "excited" : "excitable",
    'hangover" feeling after night' : "hangover feeling after night",
    "concentration" : "concentration problems",
    "unusual sense of well" : "unusual sense of wellbeing",
    "flushing (warmth" : "hot flushes",
    "flushing (sudden warmth" : "hot flushes",
    "heat flushing" : "hot flushes",
    "hyperactive (mentally" : "hyperactive",
    "have thoughts about suicide" : "suicidal thoughts",
    "thoughts of suicide" : "suicidal thoughts",
    "blood in stool stools" : "blood in stool",
    "hive" : "hives",
    "stomach pain (upper right side" : "stomach pain",
    "feelings of low self" : "feelings of low self esteem",
    "esteem" : "feelings of low self esteem",
    "breath that smells like fruit" : "fruity breath odor",
    "more hungry" : "increased hunger",
    "hostile" : "hostility",
    "runny" : "runny nose",
    "not hungry" : "appetite loss",
    "mood" : "mood changes",
    "side effects such as dry mouth" : "dry mouth",
    "feeling anxious" : "anxiety",
    "throwing up" : "vomiting",
    "like passing out" : "lightheadedness",
    "sleep problems ( insomnia" : "insomnia",
    "like symptoms" : "flu like symptoms",
    "feeling light" : "lightheadedness",
    "tremor" : "tremors",
    "trouble concentrating" : "concentration problems",
    "tired" : "tiredness"
}

df["side_effects"] = df["side_effects"].replace(synonym_map)

STOP_WORDS = {"lips", "tongue", "face", "throat", "your", "reaction", "emergency", "feet",
              "redness", "swelling", "eyes", "notice", "professional", "healthcare", "american",
              "instructions", "bothersome", "buzzing", "place", "thickening", "shoulder", "breast",
              "waist", "eyelid", "crawling", "ankles", "walking", "eating", "aggressive", "increased",
              "women", "physically", "appetite", "occasionally", "arms", "legs", "dextromethorphan", "desire",
              "breasts", "arm", "leg", "foot", "hand", "hands", "serious", "hydrochlorothiazide", "irregular",
              "white", "back", "mouth", "nose", "pharmacist", "doctor", "nervous", "snapping", "balance", "blue",
              "fingernail", "fingernails", "needles", "memory", "scaly", "sometimes", "rarely", "muscle", "muscles",
              "toes", "toe", "drive", "fast", "clay", "loss", "stomach", "peeling", "severe", "headed", "lower legs",
              "burning", "pain", "coordination"}
df = df[
    (~df["side_effects"].isin(STOP_WORDS)) &
    (df["side_effects"].str.len() > 3)
]

df = df[~df["side_effects"].apply(is_junk)]

df = df.drop_duplicates(subset=["drug_name", "side_effects"]).reset_index(drop=True)

top_side_effects = df["side_effects"].value_counts().head(10)
print(f"Number of total side effects: {df.shape[0]}")
print("Top Side Effects:")
print(top_side_effects)
print("\nFirst 15 Cleaned Rows:")
print(df["side_effects"].head(15))

df.to_csv("long_side_effects.csv", index=False)