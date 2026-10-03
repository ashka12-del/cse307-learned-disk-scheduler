"""Request-window features available before a scheduling decision."""

from __future__ import annotations

import numpy as np

BASIC_FEATURE_COLUMNS = [
    "variance", "range", "mean_absolute_gap", "gap_cv", "direction_consistency",
    "mean_distance_from_head", "fraction_below_head", "fraction_above_head",
]

EXTENDED_FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS + [
    "trend_correlation", "fraction_small_gaps", "median_gap", "max_gap",
]

# The main, interpretable decision tree intentionally keeps the original features.
FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS


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
    indices = np.arange(values.size, dtype=float)
    if values.size > 1 and values.std() > 0:
        trend_correlation = float(abs(np.corrcoef(indices, values)[0, 1]))
    else:
        trend_correlation = 0.0
    return {
        "variance": float(values.var()),
        "range": float(values.max() - values.min()),
        "mean_absolute_gap": mean_gap,
        "gap_cv": gap_cv,
        "direction_consistency": direction_consistency,
        "mean_distance_from_head": float(np.abs(values - head).mean()),
        "fraction_below_head": float((values < head).mean()),
        "fraction_above_head": float((values > head).mean()),
        "trend_correlation": trend_correlation,
        "fraction_small_gaps": float((absolute_gaps <= 5).mean()) if absolute_gaps.size else 1.0,
        "median_gap": float(np.median(absolute_gaps)) if absolute_gaps.size else 0.0,
        "max_gap": float(absolute_gaps.max()) if absolute_gaps.size else 0.0,
    }

