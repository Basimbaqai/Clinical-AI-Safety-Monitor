"""
models/evidence.py
==================
Data classes and enums for the Knowledge Verifier stage (Stage 2) of CASM.
"""

from dataclasses import dataclass, field
from enum import Enum


class NLIVerdict(str, Enum):
    """Outcome of the NLI aggregation step for a single claim."""
    SUPPORTED    = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    AMBIGUOUS    = "AMBIGUOUS"


@dataclass
class SearchResult:
    """
    A single PubMed abstract retrieved from the Chroma vector store.

    Attributes
    ----------
    pmid        : PubMed identifier string (e.g. "38123456").
    abstract    : Full abstract text used as NLI premise.
    title       : Article title for display / logging.
    distance    : Cosine distance from the query vector (lower = more similar).
    drug_filter : Drug name used to narrow this fetch, if any.
    """
    pmid:        str
    abstract:    str
    title:       str   = ""
    distance:    float = 0.0
    drug_filter: str   = ""


@dataclass
class EvidenceResult:
    """
    Structured verification output for a single Claim.

    Attributes
    ----------
    claim_id       : Mirrors Claim.claim_id.
    claim_text     : Original claim text (for display).
    verdict        : Aggregated NLI verdict across all retrieved abstracts.
    confidence     : Weighted mean confidence of the majority-verdict abstracts.
    pubmed_hits    : Top-K SearchResult objects used for NLI.
    fda_count      : Total adverse event reports for the primary drug on openFDA.
    fda_serious    : Subset of fda_count flagged as serious outcomes.
    nli_scores     : Per-abstract NLI confidence scores (same order as pubmed_hits).
    nli_verdicts   : Per-abstract NLI verdict labels.
    skipped        : True when requires_verification was False (claim not checked).
    error          : Non-empty string when verification failed with an exception.
    """
    claim_id:     str
    claim_text:   str
    verdict:      NLIVerdict         = NLIVerdict.AMBIGUOUS
    confidence:   float              = 0.0
    pubmed_hits:  list[SearchResult] = field(default_factory=list)
    fda_count:    int                = 0
    fda_serious:  int                = 0
    nli_scores:   list[float]        = field(default_factory=list)
    nli_verdicts: list[str]          = field(default_factory=list)
    skipped:      bool               = False
    error:        str                = ""

    def to_dict(self) -> dict:
        return {
            "claim_id":     self.claim_id,
            "claim_text":   self.claim_text,
            "verdict":      self.verdict.value,
            "confidence":   round(self.confidence, 4),
            "fda_count":    self.fda_count,
            "fda_serious":  self.fda_serious,
            "nli_scores":   [round(s, 4) for s in self.nli_scores],
            "nli_verdicts": self.nli_verdicts,
            "pubmed_hits":  [
                {"pmid": h.pmid, "title": h.title, "distance": round(h.distance, 4)}
                for h in self.pubmed_hits
            ],
            "skipped": self.skipped,
            "error":   self.error,
        }