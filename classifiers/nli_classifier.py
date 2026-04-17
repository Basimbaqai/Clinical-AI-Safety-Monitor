"""
classifiers/nli_classifier.py
==============================
Thin wrapper around the BiomedNLP-PubMedBERT MNLI model.

Premise   = PubMed abstract text (evidence)
Hypothesis = Clinical claim text (what we are verifying)

The model is loaded once at construction time and held on GPU if available.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.evidence import NLIVerdict
from utils.constants import NLI_LABEL_MAP, NLI_MODEL_NAME

# Internal NLI label → NLIVerdict
_VERDICT_MAP: dict[str, NLIVerdict] = {
    "entailment":    NLIVerdict.SUPPORTED,
    "neutral":       NLIVerdict.AMBIGUOUS,
    "contradiction": NLIVerdict.CONTRADICTED,
}


class NLIClassifier:
    """
    BiomedNLP-PubMedBERT MNLI wrapper for clinical claim verification.

    Parameters
    ----------
    model_name : HuggingFace model identifier.
    """

    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NLIClassifier] Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[NLIClassifier] Ready — {param_count:,} params")

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, premise: str, hypothesis: str) -> tuple[NLIVerdict, float]:
        """
        Run NLI inference on a single (premise, hypothesis) pair.

        Parameters
        ----------
        premise    : PubMed abstract text (evidence).
        hypothesis : Clinical claim text (what we are checking).

        Returns
        -------
        (verdict, confidence)
            verdict    : NLIVerdict enum value.
            confidence : Softmax probability of the winning label [0, 1].
        """
        inputs = self.tokenizer.encode(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs).logits

        probs      = logits.softmax(dim=-1).cpu().numpy()[0]
        label_idx  = int(np.argmax(probs))
        label_str  = NLI_LABEL_MAP[label_idx]
        confidence = float(probs[label_idx])
        verdict    = _VERDICT_MAP[label_str]
        return verdict, confidence

    def classify_batch(
        self, premise: str, hypotheses: list[str]
    ) -> list[tuple[NLIVerdict, float]]:
        """
        Classify a single premise against multiple hypotheses.

        Calls :meth:`classify` sequentially; batching is left as a future
        optimisation.
        """
        return [self.classify(premise, h) for h in hypotheses]