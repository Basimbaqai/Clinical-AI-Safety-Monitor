"""
models/claim.py
===============
Data classes and enums for the Claim Extractor stage (Stage 1) of CASM.
"""

from dataclasses import dataclass
from enum import Enum


class ClaimType(str, Enum):
    """
    Classifies the medical nature of an extracted claim.

    Values
    ------
    DOSAGE_CLAIM      : Claim about drug dose, frequency, or administration route.
    DRUG_SAFETY_CLAIM : Claim about drug hazards, contraindications, or warnings.
    DRUG_INTERACTION  : Claim about interactions between two or more drugs.
    PROCEDURAL_CLAIM  : Claim about monitoring, lab tests, or follow-up actions.
    DIAGNOSIS_CLAIM   : Claim about a patient diagnosis or clinical presentation.
    POPULATION_CLAIM  : Claim specific to a patient sub-population (e.g. elderly,
                        pregnant, renal-impaired).
    GENERAL_MEDICAL   : Fallback for claims with medical content that does not
                        fit a more specific category.
    """
    DOSAGE_CLAIM      = "DOSAGE_CLAIM"
    DRUG_SAFETY_CLAIM = "DRUG_SAFETY_CLAIM"
    DRUG_INTERACTION  = "DRUG_INTERACTION"
    PROCEDURAL_CLAIM  = "PROCEDURAL_CLAIM"
    DIAGNOSIS_CLAIM   = "DIAGNOSIS_CLAIM"
    POPULATION_CLAIM  = "POPULATION_CLAIM"
    GENERAL_MEDICAL   = "GENERAL_MEDICAL"


@dataclass
class Entity:
    """
    A single named entity recognised within a claim sentence.

    Attributes
    ----------
    text  : The surface form of the entity as it appears in the sentence.
    label : The NER label assigned by the pipeline that detected it.
              Med7 labels  : DRUG | DOSAGE | STRENGTH | FORM | FREQUENCY | ROUTE | DURATION
              scispaCy labels: DISEASE | CONDITION | DISORDER
    start : Character offset (inclusive) of the entity within the sentence.
    end   : Character offset (exclusive) of the entity within the sentence.
    """
    text:  str
    label: str
    start: int
    end:   int


@dataclass
class Claim:
    """
    A single structured, atomic medical claim extracted from an LLM response.

    Attributes
    ----------
    claim_id              : Unique identifier for this claim (e.g. "claim_0").
    claim_text            : The raw sentence text from which the claim was derived.
    claim_type            : Semantic category of the claim (see ClaimType).
    entities              : All named entities detected by the hybrid pipeline.
    drug_names            : Deduplicated list of drug/medication names found in the claim.
    dosages               : Deduplicated list of dose/strength values (e.g. "200mg").
    conditions            : Deduplicated list of diseases or clinical conditions mentioned.
    requires_verification : True when the claim type demands downstream fact-checking.
    confidence            : Placeholder score in [0, 1]; populated by the Confidence
                            Calibrator stage — always 0.0 at extraction time.
    """
    claim_id:              str
    claim_text:            str
    claim_type:            ClaimType
    entities:              list[Entity]
    drug_names:            list[str]
    dosages:               list[str]
    conditions:            list[str]
    requires_verification: bool
    confidence:            float