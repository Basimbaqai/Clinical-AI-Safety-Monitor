"""
utils/aggregation.py
====================
NLI verdict aggregation logic for the Knowledge Verifier stage.
"""

from collections import Counter

import numpy as np

from models.evidence import NLIVerdict


def aggregate_verdicts(
    verdicts:    list[NLIVerdict],
    confidences: list[float],
) -> tuple[NLIVerdict, float]:
    """
    Combine per-abstract NLI results into a single claim-level verdict.

    Rules
    -----
    1. Majority label wins.
    2. In a tie between CONTRADICTED and AMBIGUOUS, CONTRADICTED wins
       (conservative safety bias).
    3. In a tie between SUPPORTED and AMBIGUOUS, AMBIGUOUS wins.
    4. The returned confidence is the weighted mean of confidences for the
       winning-verdict abstracts.

    Parameters
    ----------
    verdicts    : List of NLIVerdict values, one per abstract.
    confidences : Parallel list of confidence scores.

    Returns
    -------
    (aggregate_verdict, mean_confidence)
    """
    if not verdicts:
        return NLIVerdict.AMBIGUOUS, 0.0

    counts    = Counter(verdicts)
    max_votes = max(counts.values())

    # Priority for ties: CONTRADICTED > AMBIGUOUS > SUPPORTED (safety-first)
    priority = [NLIVerdict.CONTRADICTED, NLIVerdict.AMBIGUOUS, NLIVerdict.SUPPORTED]
    winner   = next(v for v in priority if counts.get(v, 0) == max_votes)

    winning_confs = [c for v, c in zip(verdicts, confidences) if v == winner]
    mean_conf     = float(np.mean(winning_confs)) if winning_confs else 0.0

    return winner, mean_conf