"""
pipeline/claim_extractor.py
============================
CASM — Claim Extractor (Hybrid Ensemble)
=========================================
Stage 1 of the CASM (Clinical Automated Safety Monitor) pipeline.

Single responsibility: convert a raw LLM clinical response into a list of
structured, typed, and entity-tagged atomic claims.

Dual-pipeline architecture:
  - Med7 (en_core_med7_trf)  → DRUG, DOSAGE, STRENGTH, FORM, FREQUENCY, ROUTE, DURATION
  - scispaCy (en_core_sci_md) → DISEASE, fallback conditions

Patient context and population flags are handled downstream by the
Knowledge Verifier — not here.
"""

import re
from typing import Optional

import spacy
import torch

from models.claim import Claim, ClaimType, Entity
from utils.constants import (
    CLAIM_TYPE_KEYWORDS,
    CONDITION_ABBREV_REGEX,
    DOSAGE_REGEX,
    DRUG_SUFFIX_REGEX,
    MED7_DOSAGE_LABELS,
    MED7_DRUG_LABELS,
    SCI_DISEASE_LABELS,
    SPLIT_CONJUNCTIONS,
)


class ClaimExtractor:
    """
    Hybrid Ensemble Claim Extractor.

    Converts a raw LLM clinical response into a list of structured Claim
    objects by running two complementary NLP pipelines:

    Pipeline 1 — Med7 (en_core_med7_trf):
        Handles DRUG, DOSAGE, STRENGTH, FORM, FREQUENCY, ROUTE, DURATION.
        High precision for clinical instructions.
        Transformer-based — benefits significantly from GPU acceleration.

    Pipeline 2 — scispaCy (en_core_sci_md):
        Fallback for DISEASE/CONDITION detection.
        Only disease-like entities are kept; generic ENTITY noise is discarded.

    Parameters
    ----------
    med7_model  : spaCy model name for the Med7 transformer pipeline.
    sci_model   : spaCy model name for the scispaCy pipeline.
    require_gpu : If True, raises RuntimeError when no CUDA GPU is detected.
    """

    def __init__(
        self,
        med7_model:  str  = "en_core_med7_trf",
        sci_model:   str  = "en_core_sci_md",
        require_gpu: bool = False,
    ):
        self._setup_device(require_gpu=require_gpu)

        print(f"[ClaimExtractor] Loading Med7 model    : {med7_model}")
        self.nlp_med7 = spacy.load(med7_model)

        print(f"[ClaimExtractor] Loading scispaCy model: {sci_model}")
        self.nlp_sci  = spacy.load(sci_model)

        print("[ClaimExtractor] Hybrid ensemble ready.")

    # ── Device Setup ──────────────────────────────────────────────────────────

    @staticmethod
    def _setup_device(require_gpu: bool = False) -> None:
        """
        Configure spaCy's compute device before any model is loaded.

        Must be called BEFORE spacy.load() — spaCy allocates model tensors
        at load time and will not migrate them afterwards.

        Priority: CUDA GPU → CPU (silent fallback unless require_gpu=True).
        """
        if torch.cuda.is_available():
            activated = spacy.prefer_gpu()
            gpu_name  = torch.cuda.get_device_name(0)
            vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
            status    = "ON" if activated else "OFF (model version mismatch — see docstring)"
            print(
                f"[ClaimExtractor] GPU detected : {gpu_name} "
                f"({vram_gb:.1f} GB VRAM) — spaCy GPU={status}"
            )
        else:
            if require_gpu:
                raise RuntimeError(
                    "[ClaimExtractor] require_gpu=True but no CUDA GPU found. "
                    "Install CUDA + torch GPU build, or set require_gpu=False."
                )
            print("[ClaimExtractor] No GPU detected — running on CPU.")

    @staticmethod
    def device_info() -> dict:
        """Return a summary of the current compute environment."""
        return {
            "cuda_available":    torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_vram_gb":      round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                                 if torch.cuda.is_available() else None,
            "spacy_gpu_active":  spacy.prefer_gpu() if torch.cuda.is_available() else False,
            "torch_version":     torch.__version__,
            "spacy_version":     spacy.__version__,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, llm_response: str) -> list[Claim]:
        """
        Extract a list of Claim objects from raw LLM output text.

        The text is first split into atomic sentences, then each sentence is
        processed independently through the hybrid NLP pipeline. Sentences
        with no detectable medical content are silently dropped.

        Parameters
        ----------
        llm_response : Raw text produced by the LLM.

        Returns
        -------
        List of Claim objects, one per extracted sentence.
        """
        sentences = self._split_into_sentences(llm_response)
        claims = []
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            claim = self._process_sentence(sentence, idx)
            if claim:
                claims.append(claim)
        print(f"[ClaimExtractor] {len(claims)} claim(s) extracted.")
        return claims

    def extract_as_dict(self, llm_response: str) -> list[dict]:
        """
        Extract claims and return them as a list of JSON-serialisable dicts.

        Convenience wrapper around :meth:`extract`.
        """
        return [self._to_dict(c) for c in self.extract(llm_response)]

    # ── Sentence Splitting ────────────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split raw text into a flat list of atomic sentence strings.

        Two-stage:
          1. scispaCy sentence boundary detection.
          2. :meth:`_split_on_conjunctions` further splits compound sentences.

        Sentences shorter than four words are discarded as fragments.
        """
        doc       = self.nlp_sci(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        final     = []
        for sentence in sentences:
            final.extend(self._split_on_conjunctions(sentence))
        return [s for s in final if len(s.split()) >= 4]

    def _split_on_conjunctions(self, sentence: str) -> list[str]:
        """
        Split a single sentence on clinical conjunctions from SPLIT_CONJUNCTIONS.

        Each pattern is applied iteratively so multiple conjunctions are handled.
        """
        parts = [sentence]
        for pattern in SPLIT_CONJUNCTIONS:
            new_parts = []
            for part in parts:
                split = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend([s.strip() for s in split if s.strip()])
            parts = new_parts
        return parts

    # ── Claim Processing ──────────────────────────────────────────────────────

    def _process_sentence(self, sentence: str, idx: int) -> Optional[Claim]:
        """
        Run the hybrid NLP pipeline on a single sentence and return a Claim,
        or None if the sentence contains no medical content.

        Steps:
          1. Med7 identifies drugs, dosages, and other clinical structure.
          2. scispaCy identifies diseases/conditions not covered by Med7.
          3. Both entity lists are merged (Med7 is authoritative).
          4. Derived fields (drug_names, dosages, conditions) are extracted.
          5. Claim type is classified via keyword scoring.
        """
        # Pipeline 1: Med7
        doc_med7      = self.nlp_med7(sentence)
        med7_entities = [
            Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            for ent in doc_med7.ents
        ]

        # Pipeline 2: scispaCy — disease/condition entities only
        doc_sci             = self.nlp_sci(sentence)
        sci_disease_entities = [
            Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            for ent in doc_sci.ents
            if ent.label_ in SCI_DISEASE_LABELS
        ]

        # Merge: Med7 authoritative; scispaCy fills disease gaps
        med7_spans      = {(e.start, e.end) for e in med7_entities}
        merged_entities = list(med7_entities)
        for e in sci_disease_entities:
            if (e.start, e.end) not in med7_spans:
                merged_entities.append(e)

        drug_names = self._extract_drugs(med7_entities, sentence)
        dosages    = self._extract_dosages(med7_entities, sentence)
        conditions = self._extract_conditions(sci_disease_entities, sentence)
        claim_type = self._classify_claim_type(sentence, merged_entities)

        # Drop sentences with no medical content
        if claim_type == ClaimType.GENERAL_MEDICAL and not merged_entities:
            return None

        requires_verification = claim_type in {
            ClaimType.DOSAGE_CLAIM,
            ClaimType.DRUG_SAFETY_CLAIM,
            ClaimType.DRUG_INTERACTION,
            ClaimType.POPULATION_CLAIM,
        }

        return Claim(
            claim_id=f"claim_{idx}",
            claim_text=sentence,
            claim_type=claim_type,
            entities=merged_entities,
            drug_names=drug_names,
            dosages=dosages,
            conditions=conditions,
            requires_verification=requires_verification,
            confidence=0.0,
        )

    # ── Claim Type Classification ─────────────────────────────────────────────

    def _classify_claim_type(
        self, sentence: str, entities: list[Entity]
    ) -> ClaimType:
        """
        Assign a ClaimType to a sentence using keyword frequency scoring.

        The type with the most keyword matches wins. If no keywords match,
        ClaimType.GENERAL_MEDICAL is returned.
        """
        sentence_lower = sentence.lower()
        scores: dict[ClaimType, int] = {ct: 0 for ct in ClaimType}

        for claim_type, keywords in CLAIM_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    scores[claim_type] += 1

        best = max(scores, key=lambda ct: scores[ct])
        return best if scores[best] > 0 else ClaimType.GENERAL_MEDICAL

    # ── Entity Extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _clean_drug_name(text: str) -> str:
        """
        Normalise a raw Med7 drug span by removing common extraction artefacts.

        Handles parenthetical abbreviations (e.g. "Fluorometholone (FML)")
        and strips leading/trailing non-alphanumeric characters.
        """
        text = re.sub(r'\s*\(.*?\)\s*$', '', text)
        text = re.sub(r'^[^A-Za-z]+|[^A-Za-z0-9]+$', '', text)
        return text.strip()

    def _extract_drugs(
        self, med7_entities: list[Entity], sentence: str
    ) -> list[str]:
        """
        Build a deduplicated list of drug names from Med7 entities and a
        regex suffix fallback.
        """
        drugs = [
            self._clean_drug_name(e.text)
            for e in med7_entities
            if e.label in MED7_DRUG_LABELS
        ]
        drugs = [d for d in drugs if d]

        for d in re.findall(DRUG_SUFFIX_REGEX, sentence):
            if d not in drugs:
                drugs.append(d)

        return list(dict.fromkeys(drugs))

    def _extract_dosages(
        self, med7_entities: list[Entity], sentence: str
    ) -> list[str]:
        """
        Build a deduplicated list of dose/strength values from Med7 entities
        and a regex numeric fallback.
        """
        dosages = [e.text for e in med7_entities if e.label in MED7_DOSAGE_LABELS]

        for d in re.findall(DOSAGE_REGEX, sentence, re.IGNORECASE):
            if d.strip() not in dosages:
                dosages.append(d.strip())

        return list(dict.fromkeys(dosages))

    def _extract_conditions(
        self, sci_entities: list[Entity], sentence: str
    ) -> list[str]:
        """
        Build a deduplicated list of disease/condition names from scispaCy
        entities and a known-abbreviation fallback.
        """
        conditions = [e.text for e in sci_entities]

        for a in re.findall(CONDITION_ABBREV_REGEX, sentence):
            if a not in conditions:
                conditions.append(a)

        return list(dict.fromkeys(conditions))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _to_dict(self, claim: Claim) -> dict:
        """Serialise a Claim to a plain, JSON-serialisable dictionary."""
        return {
            "claim_id":              claim.claim_id,
            "claim_text":            claim.claim_text,
            "claim_type":            claim.claim_type.value,
            "entities":              [
                {"text": e.text, "label": e.label} for e in claim.entities
            ],
            "drug_names":            claim.drug_names,
            "dosages":               claim.dosages,
            "conditions":            claim.conditions,
            "requires_verification": claim.requires_verification,
            "confidence":            claim.confidence,
        }