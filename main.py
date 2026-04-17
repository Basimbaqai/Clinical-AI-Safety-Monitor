"""
main.py
=======
CASM — Clinical Automated Safety Monitor
Top-level entry point.

Wires up all pipeline components and runs the full two-stage flow:
  Stage 1 — ClaimExtractor  : LLM response → structured Claim objects
  Stage 2 — KnowledgeVerifier : Claim → EvidenceResult (NLI + PubMed + FDA)

Usage
-----
    python main.py
    python main.py --response "Prescribe Metformin 500mg twice daily."
    python main.py --response "..." --out results.json

Arguments
---------
--response TEXT   Raw LLM clinical response to analyse.
                  Defaults to the three built-in demo scenarios.
--out PATH        Path for JSON output (default: outputs/results.json).
--require-gpu     Raise an error if no CUDA GPU is found (default: False).
--top-k INT       Number of PubMed abstracts to retrieve per claim (default: 5).
--min-hits INT    Min Chroma hits before triggering live PubMed fetch (default: 2).
"""

import argparse
import json
import os
import sys

# ── Ensure project root is on sys.path when run directly ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from clients.chroma_store import ChromaStore
from clients.openfda_client import OpenFDAClient
from clients.pubmed_fetcher import PubMedFetcher
from classifiers.nli_classifier import NLIClassifier
from pipeline.claim_extractor import ClaimExtractor
from pipeline.knowledge_verifier import KnowledgeVerifier


# ── Demo scenarios (used when no --response is supplied) ─────────────────────

DEMO_RESPONSES = [
    (
        "Scenario A — Corneal haze / FML",
        "To treat corneal haze 5 months post-TransPRK, prescribe Fluorometholone (FML) "
        "0.1% eye drops four times daily for 4 weeks. You may also consider Oral Vitamin C "
        "(1000mg/day) to support corneal healing.",
    ),
    (
        "Scenario B — UTI treatment",
        "For a patient with a confirmed UTI, prescribe Trimethoprim 200mg twice daily for "
        "7 days. If symptoms persist after 48 hours, switch to Nitrofurantoin 100mg four "
        "times daily for 5 days. Monitor renal function with creatinine levels weekly.",
    ),
    (
        "Scenario C — T2DM / CKD safety",
        "In a T2DM patient with CKD stage 3, avoid Metformin 500mg if eGFR drops below "
        "30 ml/min due to risk of lactic acidosis. Consider switching to Sitagliptin 50mg "
        "once daily, which is renally dosed and safer in this population. Monitor HbA1c "
        "every 3 months.",
    ),
]

VERDICT_ICON = {
    "SUPPORTED":    "✅",
    "CONTRADICTED": "❌",
    "AMBIGUOUS":    "⚠️ ",
}


# ── Result Display ────────────────────────────────────────────────────────────

def print_results(results, title: str = "") -> None:
    """Pretty-print a list of EvidenceResult objects."""
    if title:
        print(f"\n{'═' * 72}")
        print(f"  {title}")
        print(f"{'═' * 72}")

    for r in results:
        icon = "⏭️ " if r.skipped else VERDICT_ICON.get(r.verdict.value, "?")
        print(f"\n[{r.claim_id}] {icon} {r.verdict.value if not r.skipped else 'SKIPPED'}")
        print(f"  Claim     : {r.claim_text[:90]}")

        if r.skipped:
            print("  (no verification required for this claim type)")
            continue
        if r.error:
            print(f"  ERROR     : {r.error}")
            continue

        print(f"  Confidence: {r.confidence:.3f}")
        print(f"  FDA events: total={r.fda_count:,}  serious={r.fda_serious:,}")
        print(f"  Abstracts : {len(r.pubmed_hits)} used")

        for i, (hit, verdict, score) in enumerate(
            zip(r.pubmed_hits, r.nli_verdicts, r.nli_scores)
        ):
            v_icon = VERDICT_ICON.get(verdict, "?")
            print(
                f"    [{i+1}] pmid={hit.pmid}  {v_icon} {verdict}"
                f"  conf={score:.3f}  dist={hit.distance:.3f}"
            )
            if hit.title:
                print(f"         title: {hit.title[:75]}")


# ── Pipeline Bootstrap ────────────────────────────────────────────────────────

def build_pipeline(args: argparse.Namespace):
    """Instantiate all pipeline components and return (extractor, verifier)."""
    print("\n[CASM] Initialising pipeline components…")

    extractor = ClaimExtractor(require_gpu=args.require_gpu)

    chroma  = ChromaStore()
    nli     = NLIClassifier()
    fda     = OpenFDAClient()
    pubmed  = PubMedFetcher(store=chroma)

    verifier = KnowledgeVerifier(
        chroma_store   = chroma,
        nli_classifier = nli,
        fda_client     = fda,
        pubmed_fetcher = pubmed,
        top_k          = args.top_k,
        min_hits       = args.min_hits,
    )

    print("[CASM] Pipeline ready.\n")
    return extractor, verifier


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CASM — Clinical Automated Safety Monitor"
    )
    parser.add_argument(
        "--response", type=str, default=None,
        help="Raw LLM clinical response to analyse. Defaults to built-in demo scenarios.",
    )
    parser.add_argument(
        "--out", type=str, default="outputs/results.json",
        help="Path for JSON output (default: outputs/results.json).",
    )
    parser.add_argument(
        "--require-gpu", action="store_true", default=False,
        help="Raise an error if no CUDA GPU is found.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of PubMed abstracts to retrieve per claim (default: 5).",
    )
    parser.add_argument(
        "--min-hits", type=int, default=2,
        help="Min Chroma hits before live PubMed fetch (default: 2).",
    )
    args = parser.parse_args()

    extractor, verifier = build_pipeline(args)

    all_output: dict[str, list[dict]] = {}

    if args.response:
        # ── Single response mode ───────────────────────────────────────────
        print(f"[CASM] Analysing response:\n  {args.response[:120]}\n")
        claims  = extractor.extract(args.response)
        results = verifier.verify_all(claims)
        print_results(results, title="CASM Results")
        all_output["custom"] = [r.to_dict() for r in results]

    else:
        # ── Demo mode: run all three built-in scenarios ────────────────────
        for label, response in DEMO_RESPONSES:
            print(f"\n{'─' * 72}")
            print(f"  {label}")
            print(f"{'─' * 72}")
            print(f"  Input: {response[:100]}…")
            claims  = extractor.extract(response)
            results = verifier.verify_all(claims)
            print_results(results, title=label)
            key = label.split("—")[0].strip().lower().replace(" ", "_")
            all_output[key] = [r.to_dict() for r in results]

    # ── Save JSON output ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"\n[CASM] Results saved → {args.out}")


if __name__ == "__main__":
    main()