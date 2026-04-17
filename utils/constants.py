"""
utils/constants.py
==================
Shared constants used across the CASM pipeline:
  - NER label sets for Med7 and scispaCy
  - Claim-type keyword map for rule-based classification
  - Conjunction patterns for sentence splitting
  - External service configuration
"""

from models.claim import ClaimType

# ── Med7 clinical labels ───────────────────────────────────────────────────────

# Entity labels recognised by Med7 as drug/medication names.
MED7_DRUG_LABELS = {"DRUG"}

# Entity labels recognised by Med7 as dose or strength values (e.g. "200mg").
MED7_DOSAGE_LABELS = {"DOSAGE", "STRENGTH"}

# Real clinical information from Med7 intentionally excluded from Claim.entities
# to keep the entity list focused on named clinical objects.
MED7_CLINICAL_NOISE = {"FREQUENCY", "DURATION", "FORM", "ROUTE"}

# scispaCy disease-like labels (generic ENTITY noise is excluded).
SCI_DISEASE_LABELS = {"DISEASE", "CONDITION", "DISORDER"}

# ── Claim-Type Keyword Map ─────────────────────────────────────────────────────

# Maps each ClaimType to trigger keywords for rule-based classification.
# The type whose keywords score the most matches in a sentence wins;
# ties default to GENERAL_MEDICAL.
CLAIM_TYPE_KEYWORDS: dict[ClaimType, list[str]] = {
    ClaimType.DOSAGE_CLAIM: [
        "mg", "mcg", "ml", "%", "dose", "dosage", "twice", "once", "three times",
        "daily", "bid", "tid", "qid", "weekly", "monthly", "prescribe",
        "administer", "give", "start", "initiate", "tablet", "capsule", "drops",
    ],
    ClaimType.DRUG_SAFETY_CLAIM: [
        "avoid", "contraindicated", "do not use", "caution", "warning",
        "unsafe", "dangerous", "harmful", "risk", "adverse", "side effect",
        "toxicity", "nephrotoxic", "hepatotoxic", "cardiotoxic",
    ],
    ClaimType.DRUG_INTERACTION: [
        "interaction", "interacts", "combined with", "together with",
        "concurrent", "concomitant", "potentiates", "inhibits", "induces",
        "bleeding risk", "increased risk when",
    ],
    ClaimType.PROCEDURAL_CLAIM: [
        "monitor", "check", "test", "measure", "assess", "every", "weeks",
        "months", "follow up", "review", "screen", "scan", "blood test",
        "lab", "hba1c", "creatinine", "egfr", "blood pressure",
    ],
    ClaimType.DIAGNOSIS_CLAIM: [
        "diagnosis", "diagnose", "patient has", "presents with", "suffering from",
        "consistent with", "indicative of", "suggests", "confirms",
    ],
    ClaimType.POPULATION_CLAIM: [
        "elderly", "pediatric", "children", "pregnant", "renal", "kidney",
        "hepatic", "liver", "geriatric", "neonatal", "trimester",
    ],
}

# ── Sentence-Split Conjunction Patterns ───────────────────────────────────────

# Regex patterns for conjunctive phrases that often join two independent
# clinical instructions within a single sentence.
SPLIT_CONJUNCTIONS: list[str] = [
    r"\band also\b", r"\badditionally\b", r"\bfurthermore\b",
    r"\bin addition\b", r"\bmoreover\b", r"\bhowever\b", r"\balternatively\b",
]

# ── Regex Fallback Patterns ───────────────────────────────────────────────────

# Drug name suffix regex — catches brand/generic names Med7 misses.
DRUG_SUFFIX_REGEX = (
    r"\b[A-Z][a-z]+(?:in|ol|ine|ide|ate|mab|nib|pril|sartan|statin"
    r"|mycin|cillin|zole|olone)\b"
)

# Dosage numeric pattern — matches values like "200mg", "0.5 mcg/kg".
DOSAGE_REGEX = (
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|%|units?|IU|mmol)"
    r"(?:/(?:day|kg|dose))?\b"
)

# Known clinical abbreviations for condition detection.
CONDITION_ABBREV_REGEX = (
    r"\b(?:CKD|DM|T2DM|HTN|CAD|CHF|COPD|AF|DVT|PE|UTI|PRK|TransPRK)\b"
)

# ── Chroma / Embedding Configuration ─────────────────────────────────────────

CHROMA_PATH      = "./data/chroma_store"
COLLECTION_NAME  = "pubmed_abstracts"
TOP_K            = 5
MIN_HITS         = 2
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# ── PubMed / Entrez Configuration ────────────────────────────────────────────

PUBMED_MAX_FETCH = 10
ENTREZ_EMAIL     = "casm-pipeline@example.com"   # required by NCBI ToS
ENTREZ_TOOL      = "CASM-KnowledgeVerifier"
ENTREZ_SLEEP     = 0.34   # seconds between requests (NCBI 3 req/s limit)

# ── openFDA Configuration ─────────────────────────────────────────────────────

FDA_BASE    = "https://api.fda.gov/drug/event.json"
FDA_TIMEOUT = 10  # seconds

# ── NLI Configuration ─────────────────────────────────────────────────────────

NLI_MODEL_NAME = (
    "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"
)

# HuggingFace MNLI convention: 0=entailment, 1=neutral, 2=contradiction
NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}