# Clinical AI Safety Monitor (CASM)

CASM is a clinical-response safety checking project that is currently notebook-first and moving toward a packaged pipeline.

At a high level, the workflow is:
1. Extract atomic clinical claims from LLM output.
2. Route claims to external evidence sources (OpenFDA, PubMed).
3. Verify support/contradiction with deterministic logic (and optionally NLI models).

---

## Project status

This repository is in active prototyping:
- `CASM.ipynb` contains the main implemented claim extraction pipeline.
- `test.ipynb` contains an end-to-end deterministic verification prototype (OpenFDA + PubMed query and parsing logic).
- `biobert_test.ipynb` contains a biomedical NLI experiment using PubMedBERT fine-tuned for MNLI.
- `main.py` currently exists but is empty.
- `casm_verifier/` exists but is currently a scaffold (empty files/folders at the moment).
- `tests/` exists but is currently empty.

---

## Repository structure

```text
Clinical-AI-Safety-Monitor/
  CASM.ipynb
  biobert_test.ipynb
  test.ipynb
  main.py
  pyproject.toml
  requirements
  en_core_med7_trf-3.4.2.1-py3-none-any.whl
  en_core_sci_md-0.5.4.tar.gz
    
```

---

## Core components

### 1) Claim Extractor (`CASM.ipynb`)

`CASM.ipynb` implements a hybrid ensemble extractor (`ClaimExtractor`) that converts raw LLM responses into structured claim objects.

Implemented design:
- **Pipeline A (Med7, `en_core_med7_trf`)** for medication-centric entities:
  - `DRUG`, `DOSAGE`, `STRENGTH`, `FORM`, `FREQUENCY`, `ROUTE`, `DURATION`
- **Pipeline B (scispaCy, `en_core_sci_md`)** for disease/condition entities.
- **Sentence splitting** using scispaCy sentence boundaries + conjunction splitting.
- **Claim classification** via keyword scoring into:
  - `DOSAGE_CLAIM`
  - `DRUG_SAFETY_CLAIM`
  - `DRUG_INTERACTION`
  - `PROCEDURAL_CLAIM`
  - `DIAGNOSIS_CLAIM`
  - `POPULATION_CLAIM`
  - `GENERAL_MEDICAL`
- **Derived fields** extracted per claim:
  - `drug_names`, `dosages`, `conditions`
- **Output shape** (JSON serializable):
  - `claim_id`, `claim_text`, `claim_type`, `entities`, `drug_names`, `dosages`, `conditions`, `requires_verification`, `confidence`

Notes from current implementation:
- GPU is auto-detected through `torch.cuda.is_available()` and `spacy.prefer_gpu()`.
- There is an explicit note about potential Med7/spaCy version mismatch causing CPU fallback.

---

### 2) Deterministic verification prototype (`test.ipynb`)

`test.ipynb` demonstrates verifier routing concepts with strict schemas and hard-coded routes.

What is implemented in the notebook:
- Pydantic models for claims and extracted entities.
- OpenFDA query generation:
  - `generate_openfda_query(drug, age)`
  - target: adverse-event counts by reaction term.
- PubMed query generation/parsing:
  - `generate_pubmed_query(drug, condition, population)` via Entrez `esearch`
  - PMID fetch helpers (`esummary`, `efetch`)
  - XML parsing of article metadata/abstracts.
- Simple routing logic:
  - Dosage + age -> OpenFDA adverse events
  - General medical claim -> PubMed literature retrieval
- HTTP helper utilities (`requests`) with timeout/error handling examples.

This notebook is the practical reference for the future packaged `Knowledge Verifier` module.

---

### 3) Biomedical NLI experiment (`biobert_test.ipynb`)

`biobert_test.ipynb` evaluates textual entailment/contradiction/neutral classification for clinical premise-hypothesis pairs.

Implemented setup:
- Model: `lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli`
- Framework: Hugging Face `transformers` + PyTorch
- Labels:
  - `entailment`
  - `neutral`
  - `contradiction`
- Includes a 20-case benchmark loop with accuracy reporting by label.

This is currently an experiment notebook and not yet integrated into the main verifier flow.

---

## Dependencies

Current dependency declarations are split across:
- `pyproject.toml`
- `requirements`

### Declared in `pyproject.toml`
- `fastapi`
- `ipykernel`
- `numpy==1.26`
- `setuptools`
- `spacy==3.7.4`
- `spacy-transformers`
- `uvicorn`

### Declared in `requirements`
- `numpy==1.26`
- `fastapi`
- `uvicorn`
- `spacy==3.7.4`
- `chromadb`
- `spacy-transformers`
- `en_core_med7_trf-3.4.2.1-py3-none-any.whl`
- `scispacy==0.5.4`

### Also used in notebooks (install explicitly if needed)
- `torch`
- `requests`
- `transformers`

---

## Setup (Windows PowerShell)

```powershell
Set-Location "C:\Users\basim\PycharmProjects\Clinical-AI-Safety-Monitor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements
pip install torch requests transformers
pip install .\en_core_med7_trf-3.4.2.1-py3-none-any.whl
pip install .\en_core_sci_md-0.5.4.tar.gz
```

If you use `pyproject.toml`/`uv` instead of `requirements`, keep versions aligned manually for now.

---

## How to run current work

### Claim extraction (notebook)
1. Open `CASM.ipynb`.
2. Run cells top-to-bottom.
3. Use `ClaimExtractor().extract_as_dict(llm_response=...)`.

### Deterministic verification prototype (notebook)
1. Open `test.ipynb`.
2. Run the query generator + HTTP + XML parse cells.
3. Review OpenFDA reactions and parsed PubMed abstracts.

### Biomedical NLI benchmark (notebook)
1. Open `biobert_test.ipynb`.
2. Run all cells.
3. Review overall and per-label accuracy.

---

## Current architecture (as implemented)

```text
LLM response text
  -> ClaimExtractor (CASM.ipynb)
      -> structured claims (dict/dataclass)
          -> deterministic routing prototype (test.ipynb)
              -> OpenFDA adverse-event evidence
              -> PubMed abstract evidence
                  -> (planned) unified Knowledge Verifier output
                      -> (optional) NLI calibration via PubMedBERT
```

---

## Gaps and next steps

The repository already has useful building blocks, but these are still open:
- Port notebook verifier logic into Python modules under `casm_verifier/`.
- Implement a real `KnowledgeVerifier` class and route registry.
- Add end-to-end CLI or script entrypoint in `main.py`.
- Add automated tests under `tests/` for query builders, parsers, and routing.
- Unify dependency management (choose one primary manifest strategy).
- Integrate NLI experiment into verifier as optional scoring/calibration stage.

---

## Troubleshooting notes

- **NumPy compatibility:** current notes indicate spaCy/thinc may require `numpy < 2`; this repo pins `numpy==1.26`.
- **Med7 model + spaCy GPU behavior:** Med7 package version may not fully match newer spaCy internals, causing CPU fallback even when CUDA exists.
- **API/network reliability:** OpenFDA and NCBI endpoints can rate limit or timeout; keep request timeouts and retries.

---

## License / data use

No explicit license file is present in the repository yet. Add one before external distribution.

When using external APIs (OpenFDA, NCBI), follow each provider's usage policies and rate-limit guidance.
