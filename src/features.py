"""Request-window features available before a scheduling decision."""

from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = [
    "variance", "range", "mean_absolute_gap", "gap_cv", "direction_consistency",
    "mean_distance_from_head", "fraction_below_head", "fraction_above_head",
]


def extract_features(requests: list[int], head: int) -> dict[str, float]:
    values = np.asarray(requests, dtype=float)
    if values.size == 0:
        raise ValueError("a request window cannot be empty")
    gaps = np.diff(values)
    absolute_gaps = np.abs(gaps)
    mean_gap = float(absolute_gaps.mean()) if absolute_gaps.size else 0.0
    gap_cv = float(absolute_gaps.std() / mean_gap) if mean_gap > 0 else 0.0
    nonzero = gaps[gaps != 0]
    if nonzero.size:
        positive_fraction = float((nonzero > 0).mean())
        direction_consistency = max(positive_fraction, 1.0 - positive_fraction)
    else:
        direction_consistency = 1.0
    return {
        "variance": float(values.var()),
        "range": float(values.max() - values.min()),
        "mean_absolute_gap": mean_gap,
        "gap_cv": gap_cv,
        "direction_consistency": direction_consistency,
        "mean_distance_from_head": float(np.abs(values - head).mean()),
        "fraction_below_head": float((values < head).mean()),
        "fraction_above_head": float((values > head).mean()),
    }

