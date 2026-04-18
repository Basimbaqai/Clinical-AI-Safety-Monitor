"""
CASM — Clinical Automated Safety Monitor
FastAPI Server

Endpoints:
    GET  /health           — pipeline status
    GET  /                 — API info
    POST /extract          — Stage 1: extract claims from LLM response
    POST /verify           — Stage 2: verify existing claims
    POST /analyze          — Full pipeline (extract + verify)
    POST /analyze-verdict  — Concise pipeline (verdict + sources only)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
    or in notebook:
    await server.serve()
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(name)s: %(message)s"
)
logger = logging.getLogger("CASM-API")


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    llm_response: str = Field(..., description="Raw clinical text from LLM", min_length=1)

class ExtractResponse(BaseModel):
    success:     bool       = True
    claims:      List[dict] = Field(..., description="List of extracted claims")
    claim_count: int
    error:       Optional[str] = None

class VerifyRequest(BaseModel):
    claims: List[dict] = Field(..., description="Claims to verify", min_length=1)

class VerifyResponse(BaseModel):
    success:        bool       = True
    evidence:       List[dict] = Field(..., description="Verification results")
    verified_count: int
    error:          Optional[str] = None

class AnalyzeRequest(BaseModel):
    llm_response: str = Field(..., description="Raw clinical text from LLM", min_length=1)

class AnalyzeResponse(BaseModel):
    success:        bool       = True
    claims:         List[dict]
    evidence:       List[dict]
    claim_count:    int
    verified_count: int
    error:          Optional[str] = None

class ConciseVerdict(BaseModel):
    claim_text: str  = Field(..., description="The extracted clinical claim")
    verdict:    str  = Field(..., description="SUPPORTED / CONTRADICTED / AMBIGUOUS")
    sources:    list = Field(default_factory=list, description="PubMed abstracts used")

class VerdictOnlyResponse(BaseModel):
    success: bool               = True
    results: List[ConciseVerdict]
    error:   Optional[str]      = None

class HealthResponse(BaseModel):
    status:       str
    version:      str  = "1.0.0"
    models_ready: bool
    extractors:   dict = Field(default_factory=dict)


# ── Pipeline singletons ───────────────────────────────────────────────────────

_extractor = None
_verifier  = None


def init_models():
    global _extractor, _verifier
    logger.info("Initializing models...")
    try:
        from pipeline.claim_extractor    import ClaimExtractor
        from pipeline.knowledge_verifier import KnowledgeVerifier
        from clients.chroma_store        import ChromaStore
        from clients.pubmed_fetcher      import PubMedFetcher
        from clients.openfda_client      import OpenFDAClient
        from classifiers.nli_classifier  import NLIClassifier

        logger.info("Loading ClaimExtractor...")
        _extractor = ClaimExtractor()
        logger.info("✅ ClaimExtractor loaded")

        logger.info("Loading KnowledgeVerifier...")
        chroma    = ChromaStore()
        _verifier = KnowledgeVerifier(
            chroma_store   = chroma,
            nli_classifier = NLIClassifier(),
            fda_client     = OpenFDAClient(),
            pubmed_fetcher = PubMedFetcher(store=chroma),
        )
        logger.info("✅ KnowledgeVerifier loaded")
        logger.info("=" * 60)
        logger.info("✅ ALL MODELS INITIALIZED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI server starting...")
    init_models()
    logger.info("Server ready to accept requests")
    yield
    logger.info("FastAPI server shutting down... Goodbye!")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "CASM — Clinical Automated Safety Monitor",
    description = "Two-stage NLP pipeline for extracting and verifying clinical claims",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
async def root():
    return {
        "service": "CASM — Clinical Automated Safety Monitor",
        "version": "1.0.0",
        "documentation": "http://localhost:8000/docs",
        "endpoints": {
            "extract":         "POST /extract          — Extract claims only",
            "verify":          "POST /verify           — Verify existing claims",
            "analyze":         "POST /analyze          — Full pipeline (extract + verify)",
            "analyze-verdict": "POST /analyze-verdict  — Verdicts + sources only",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    return HealthResponse(
        status       = "healthy" if _extractor and _verifier else "degraded",
        models_ready = bool(_extractor and _verifier),
        extractors   = {
            "extractor_loaded": _extractor is not None,
            "verifier_loaded":  _verifier  is not None,
        }
    )


@app.post("/extract", response_model=ExtractResponse, tags=["Pipeline"])
async def extract_claims(request: ExtractRequest):
    """Extract structured claims from raw LLM clinical text."""
    if _extractor is None:
        raise HTTPException(503, detail="Extractor not loaded")
    try:
        logger.info(f"Extracting from {len(request.llm_response)} char response")
        claims = _extractor.extract_as_dict(request.llm_response)
        logger.info(f"Extracted {len(claims)} claim(s)")
        return ExtractResponse(success=True, claims=claims, claim_count=len(claims))
    except Exception as e:
        logger.exception("Extract failed")
        raise HTTPException(500, detail=f"Extraction failed: {e}")


@app.post("/verify", response_model=VerifyResponse, tags=["Pipeline"])
async def verify_claims(request: VerifyRequest):
    """Verify claims using PubMed + NLI + openFDA."""
    if _verifier is None:
        raise HTTPException(503, detail="Verifier not loaded")
    try:
        logger.info(f"Verifying {len(request.claims)} claim(s)")
        results  = _verifier.verify_all(request.claims)
        evidence = [r.to_dict() for r in results]
        logger.info(f"Verification complete — {len(evidence)} results")
        return VerifyResponse(success=True, evidence=evidence, verified_count=len(evidence))
    except Exception as e:
        logger.exception("Verify failed")
        raise HTTPException(500, detail=f"Verification failed: {e}")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Pipeline"])
async def full_pipeline(request: AnalyzeRequest):
    """Full end-to-end pipeline: extract claims then verify against evidence."""
    if _extractor is None or _verifier is None:
        raise HTTPException(503, detail="Models not loaded")
    try:
        logger.info("=" * 60)
        logger.info("FULL PIPELINE START")

        logger.info("[Stage 1] Extracting claims...")
        claims = _extractor.extract_as_dict(request.llm_response)
        logger.info(f"[Stage 1] ✅ Extracted {len(claims)} claim(s)")

        logger.info("[Stage 2] Verifying claims...")
        results  = _verifier.verify_all(claims)
        evidence = [r.to_dict() for r in results]
        logger.info(f"[Stage 2] ✅ Verified {len(evidence)} claim(s)")

        logger.info("FULL PIPELINE COMPLETE")
        logger.info("=" * 60)

        return AnalyzeResponse(
            success        = True,
            claims         = claims,
            evidence       = evidence,
            claim_count    = len(claims),
            verified_count = len(evidence),
        )
    except Exception as e:
        logger.exception("Full pipeline failed")
        raise HTTPException(500, detail=f"Pipeline failed: {e}")


@app.post("/analyze-verdict", response_model=VerdictOnlyResponse, tags=["Pipeline"])
async def analyze_verdict_only(request: AnalyzeRequest):
    """Streamlined pipeline — returns only claim text, verdict, and sources."""
    if _extractor is None or _verifier is None:
        raise HTTPException(503, detail="Models not loaded")
    try:
        logger.info("Running Verdict-Only pipeline...")
        claims  = _extractor.extract_as_dict(request.llm_response)
        results = _verifier.verify_all(claims)

        concise = []
        for claim, ev in zip(claims, results):
            ev_dict = ev.to_dict()
            sources = ev_dict.get("pubmed_hits") or ev_dict.get("sources") or []
            concise.append(ConciseVerdict(
                claim_text = claim.get("claim_text", ""),
                verdict    = ev_dict.get("verdict", "AMBIGUOUS"),
                sources    = sources,
            ))

        logger.info(f"Verdict-Only complete — {len(concise)} claim(s) processed")
        return VerdictOnlyResponse(success=True, results=concise)

    except Exception as e:
        logger.exception("Verdict-only pipeline failed")
        raise HTTPException(500, detail=f"Pipeline failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting CASM API server...")
    logger.info("📖 Docs: http://localhost:8000/docs")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=False)
    server = uvicorn.Server(config)

    # In a Jupyter notebook, use: await server.serve()
    uvicorn.run(app, host="0.0.0.0", port=8000)