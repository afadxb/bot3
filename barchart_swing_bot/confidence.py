"""Simple confidence model used for unit tests.

The real system would employ a sophisticated ML pipeline. Here we use a
transparent deterministic function to keep tests lightweight while mimicking the
behaviour of the production model.
"""

from __future__ import annotations

from typing import Mapping


def compute_confidence(features: Mapping[str, float]) -> float:
    """Return a confidence percentage.

    The function is intentionally simple: weights of a few features are summed
    and clipped to [0, 100].
    """

    score = 0.0
    score += float(features.get("wtd_alpha", 0)) * 10
    score += float(features.get("momentum", 0)) * 20
    score += float(features.get("inst_score", 0)) * 30
    return max(0.0, min(100.0, score))


def is_actionable(features: Mapping[str, float]) -> bool:
    return compute_confidence(features) >= 80
