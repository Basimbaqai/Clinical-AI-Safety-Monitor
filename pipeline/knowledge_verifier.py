"""
pipeline/knowledge_verifier.py
===============================
CASM — Knowledge Verifier
==========================
Stage 2 of the CASM pipeline.

Single responsibility: verify each Claim that has requires_verification=True
and produce an EvidenceResult containing an NLI verdict, confidence score,
supporting PubMed abstracts, and openFDA adverse-event counts.

Architecture (per verifiable claim):
  1. Encode claim text → Chroma semantic search (top-K abstracts)
  2. If Chroma returns < MIN_HITS, live-fetch from PubMed and upsert
  3. NLI (BiomedNLP-PubMedBERT MNLI) on each abstract → verdict + confidence
  4. Aggregate top-K NLI verdicts: majority wins; CONTRADICTED beats AMBIGUOUS in ties
  5. openFDA /drug/event.json → adverse-event count (parallel with NLI via thread pool)
  6. Produce EvidenceResult

Population flags are intentionally excluded from this stage.
"""

import concurrent.futures
import traceback
from typing import Optional

from classifiers.nli_classifier import NLIClassifier
from clients.chroma_store import ChromaStore
from clients.openfda_client import OpenFDAClient
from clients.pubmed_fetcher import PubMedFetcher
from models.claim import Claim
from models.evidence import EvidenceResult, NLIVerdict, SearchResult
from utils.aggregation import aggregate_verdicts
from utils.constants import MIN_HITS, TOP_K


class KnowledgeVerifier:
    """
    CASM Stage 2 — Knowledge Verifier.

    Accepts a list of Claim objects (from ClaimExtractor) and returns
    a parallel list of EvidenceResult objects.

    For claims with requires_verification=True:
      1. Encode claim → Chroma semantic search (top-K abstracts).
      2. If Chroma has too few results, live-fetch from PubMed and upsert.
      3. Run NLI on each abstract in a thread pool (NLI + FDA overlap).
      4. Aggregate verdicts → NLIVerdict + confidence.
      5. Query openFDA for adverse-event counts (runs in parallel with NLI).
      6. Return EvidenceResult.

    For claims with requires_verification=False:
      Returns a skipped EvidenceResult immediately.

    Parameters
    ----------
    chroma_store   : Shared ChromaStore instance.
    nli_classifier : Shared NLIClassifier instance.
    fda_client     : Shared OpenFDAClient instance.
    pubmed_fetcher : Shared PubMedFetcher instance.
    top_k          : Number of abstracts to retrieve per claim.
    min_hits       : Minimum Chroma hits before triggering live fetch.
    """

    def __init__(
        self,
        chroma_store:   ChromaStore,
        nli_classifier: NLIClassifier,
        fda_client:     OpenFDAClient,
        pubmed_fetcher: PubMedFetcher,
        top_k:          int = TOP_K,
        min_hits:       int = MIN_HITS,
    ):
        self.chroma   = chroma_store
        self.nli      = nli_classifier
        self.fda      = fda_client
        self.fetcher  = pubmed_fetcher
        self.top_k    = top_k
        self.min_hits = min_hits

    # ── Public API ────────────────────────────────────────────────────────────

    def verify_all(self, claims: list[Claim | dict]) -> list[EvidenceResult]:
        """
        Verify a list of Claim objects and return one EvidenceResult per claim.

        Parameters
        ----------
        claims : Output of ClaimExtractor.extract() or extract_as_dict().
                 Both Claim dataclass objects and plain dicts are accepted.

        Returns
        -------
        List of EvidenceResult objects in the same order as *claims*.
        """
        results = []
        for claim in claims:
            # Support both dataclass and dict representations
            if isinstance(claim, dict):
                claim_id   = claim["claim_id"]
                claim_text = claim["claim_text"]
                drug_names = claim.get("drug_names", [])
                req_verif  = claim.get("requires_verification", False)
            else:
                claim_id   = claim.claim_id
                claim_text = claim.claim_text
                drug_names = claim.drug_names
                req_verif  = claim.requires_verification

            print(f"\n[Verifier] {claim_id} — requires_verification={req_verif}")
            print(f"  claim: {claim_text[:100]}")

            if not req_verif:
                results.append(EvidenceResult(
                    claim_id=claim_id, claim_text=claim_text, skipped=True
                ))
                print("  → Skipped (no verification required)")
                continue

            result = self._verify_single(claim_id, claim_text, drug_names)
            results.append(result)

        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _verify_single(
        self,
        claim_id:   str,
        claim_text: str,
        drug_names: list[str],
    ) -> EvidenceResult:
        """Run the full verification pipeline for one claim."""
        primary_drug = drug_names[0] if drug_names else None

        try:
            # Step 1+2: Chroma semantic search
            hits = self.chroma.query(claim_text, self.top_k, drug_filter=primary_drug)
            print(f"  [Step 2] Chroma returned {len(hits)} hit(s)")

            # Step 3: Live PubMed fetch if not enough hits
            if len(hits) < self.min_hits and primary_drug:
                print(
                    f"  [Step 3] Too few hits ({len(hits)} < {self.min_hits})"
                    f" — fetching from PubMed"
                )
                self.fetcher.fetch_and_index(primary_drug)
                hits = self.chroma.query(claim_text, self.top_k, drug_filter=primary_drug)
                print(f"  [Step 3] After fetch: {len(hits)} hit(s)")

            if not hits:
                return EvidenceResult(
                    claim_id=claim_id, claim_text=claim_text,
                    verdict=NLIVerdict.AMBIGUOUS, confidence=0.0,
                    error="No PubMed abstracts found",
                )

            # Steps 4+5: NLI and openFDA in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                nli_future = pool.submit(self._run_nli, claim_text, hits)
                fda_future = pool.submit(self._run_fda, primary_drug)
                verdicts, scores    = nli_future.result()
                fda_total, fda_srs  = fda_future.result()

            # Step 5: Aggregate
            agg_verdict, agg_conf = aggregate_verdicts(verdicts, scores)
            print(f"  [Step 5] Verdict: {agg_verdict.value}  Confidence: {agg_conf:.3f}")
            print(f"  [Step 5] FDA total={fda_total}  serious={fda_srs}")

            return EvidenceResult(
                claim_id     = claim_id,
                claim_text   = claim_text,
                verdict      = agg_verdict,
                confidence   = agg_conf,
                pubmed_hits  = hits,
                fda_count    = fda_total,
                fda_serious  = fda_srs,
                nli_scores   = scores,
                nli_verdicts = [v.value for v in verdicts],
            )

        except Exception as exc:
            print(f"  [Verifier] ERROR: {exc}")
            traceback.print_exc()
            return EvidenceResult(
                claim_id=claim_id, claim_text=claim_text, error=str(exc)
            )

    def _run_nli(
        self,
        claim_text: str,
        hits:       list[SearchResult],
    ) -> tuple[list[NLIVerdict], list[float]]:
        """Run NLI on each abstract and return parallel verdict + score lists."""
        verdicts, scores = [], []
        for hit in hits:
            verdict, conf = self.nli.classify(
                premise    = hit.abstract,
                hypothesis = claim_text,
            )
            verdicts.append(verdict)
            scores.append(conf)
            print(f"    [NLI] pmid={hit.pmid}  {verdict.value}  conf={conf:.3f}")
        return verdicts, scores

    def _run_fda(self, drug_name: Optional[str]) -> tuple[int, int]:
        """Query openFDA; returns (0, 0) if no drug name is available."""
        if not drug_name:
            return 0, 0
        return self.fda.query(drug_name)