# Clinical AI Safety Monitor (CASM)

CASM is a clinical response verification pipeline. It takes free-text clinical advice (for example, LLM output), extracts structured medical claims, and checks those claims against literature and safety signals.

## What This Project Does

- Extracts atomic claims from raw clinical text using a hybrid Med7 + scispaCy pipeline.
- Classifies each claim type and decides whether verification is required.
- Retrieves PubMed abstracts from a local Chroma vector store (and live-fetches from PubMed when needed).
- Runs biomedical NLI (`SUPPORTED`, `CONTRADICTED`, `AMBIGUOUS`) against retrieved evidence.
- Pulls openFDA adverse-event totals and serious-event counts for the primary drug.
- Aggregates everything into structured JSON output.

## System Diagram


![System Diagram](docs/system-diagram.png)


<br><br><br><br>

## Project Structure

```text
Clinical-AI-Safety-Monitor/
  main.py
  pyproject.toml
  requirements
  README.md
  en_core_med7_trf-3.4.2.1-py3-none-any.whl
  en_core_sci_md-0.5.4.tar.gz
  classifiers/
    nli_classifier.py
  clients/
    chroma_store.py
    openfda_client.py
    pubmed_fetcher.py
  models/
    claim.py
    evidence.py
  pipeline/
    claim_extractor.py
    knowledge_verifier.py
  utils/
    aggregation.py
    constants.py
  data/
    chroma_store/
  outputs/
    results.json
  experiments/
    CASM.ipynb
    test.ipynb
    biobert_test.ipynb
```

## End-to-End Flow

1. Input text is passed into `pipeline/claim_extractor.py`.
2. Claims are extracted, typed, and marked with `requires_verification`.
3. Verifiable claims go to `pipeline/knowledge_verifier.py`.
4. Verifier queries `clients/chroma_store.py` for similar PubMed abstracts.
5. If hits are below threshold, `clients/pubmed_fetcher.py` fetches and indexes fresh abstracts.
6. `classifiers/nli_classifier.py` scores evidence against each claim.
7. `clients/openfda_client.py` returns adverse-event totals for the drug.
8. Verdicts and confidence are aggregated and exported as JSON by `main.py`.

## Key Modules

- `main.py`: CLI entrypoint; runs demo scenarios or one custom response and saves JSON output.
- `pipeline/claim_extractor.py`: Hybrid extractor (Med7 + scispaCy), sentence splitting, claim typing.
- `pipeline/knowledge_verifier.py`: Verification orchestration (Chroma -> PubMed fallback -> NLI + openFDA -> aggregation).
- `clients/chroma_store.py`: Embedding + semantic retrieval over persisted Chroma collection.
- `clients/pubmed_fetcher.py`: NCBI E-utilities search/fetch and Chroma upsert.
- `clients/openfda_client.py`: openFDA adverse event count queries.
- `classifiers/nli_classifier.py`: PubMedBERT MNLI wrapper.
- `models/claim.py`, `models/evidence.py`: Dataclasses/enums for pipeline IO.
- `utils/constants.py`: Shared configuration/constants (paths, model names, API settings).

## Installation

The repository includes both `requirements` and `pyproject.toml`.

### Option A (recommended): pip + requirements

```powershell
Set-Location "C:\Users\basim\PycharmProjects\Clinical-AI-Safety-Monitor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements
pip install .\en_core_sci_md-0.5.4.tar.gz
```

### Option B: uv + pyproject

```powershell
Set-Location "C:\Users\basim\PycharmProjects\Clinical-AI-Safety-Monitor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install uv
uv sync
pip install chromadb scispacy==0.5.4
pip install .\en_core_med7_trf-3.4.2.1-py3-none-any.whl
pip install .\en_core_sci_md-0.5.4.tar.gz
```

## Run CASM

### Demo mode (runs built-in scenarios)

```powershell
python main.py
```

### Single custom response

```powershell
python main.py --response "Prescribe Metformin 500mg twice daily." --out outputs\results.json
```

### Useful flags

- `--require-gpu`: fail fast if CUDA is unavailable.
- `--top-k`: number of PubMed abstracts retrieved per claim.
- `--min-hits`: minimum Chroma hits before triggering live PubMed fetch.

## Output

By default, results are written to `outputs/results.json`.

High-level output structure:

```json
{
  "custom": [
    {
      "claim_id": "claim_0",
      "claim_text": "...",
      "verdict": "SUPPORTED",
      "confidence": 0.91,
      "fda_count": 12345,
      "fda_serious": 678,
      "nli_scores": [0.9, 0.8],
      "nli_verdicts": ["SUPPORTED", "AMBIGUOUS"],
      "pubmed_hits": [{"pmid": "...", "title": "...", "distance": 0.12}],
      "skipped": false,
      "error": ""
    }
  ]
}
```

## Data and External Services

- **ChromaDB**: local persistent vector store under `data/chroma_store/`.
- **PubMed (NCBI E-utilities)**: used for live abstract retrieval.
- **openFDA**: used for adverse-event counts.
- **Hugging Face models**:
  - Sentence embedding model: `pritamdeka/S-PubMedBert-MS-MARCO`
  - NLI model: `lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli`

## Notes and Troubleshooting

- Keep `numpy==1.26` to avoid spaCy/thinc compatibility issues.
- First run can be slow due to model downloads and index loading.
- If no GPU is detected, the pipeline runs on CPU unless `--require-gpu` is set.
- Network/API rate limits can affect PubMed/openFDA requests.
- If you set `--out results.json`, ensure directory handling in your local `main.py` supports no parent folder.

## Notebook Assets

The `experiments/` notebooks are still useful for experimentation:

- `experiments/CASM.ipynb`: extraction experiments.
- `experiments/test.ipynb`: deterministic routing and API experiments.
- `experiments/biobert_test.ipynb`: NLI model behavior checks.

## License

No license file is currently present. Add a `LICENSE` before external distribution.
