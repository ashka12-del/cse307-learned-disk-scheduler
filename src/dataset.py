"""Generate labeled, non-overlapping request windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import extract_features
from schedulers import SCHEDULERS, compare_schedulers
from workloads import GENERATORS, generate_workload


def label_window(requests: list[int], head: int) -> tuple[dict[str, int], list[str]]:
    seeks = compare_schedulers(requests, head)
    minimum = min(seeks.values())
    winners = [name for name in SCHEDULERS if seeks[name] == minimum]
    return seeks, winners


def build_dataset(seeds: range, windows_per_workload: int = 20, window_size: int = 50, head: int = 50, split: str = "train") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for workload in GENERATORS:
            for window_index in range(windows_per_workload):
                requests = generate_workload(workload, rng, window_size)
                seeks, winners = label_window(requests, head)
                rows.append({
                    "split": split, "seed": seed, "workload": workload, "window_index": window_index,
                    "head": head, "requests": " ".join(map(str, requests)), **extract_features(requests, head),
                    **{f"seek_{name}": value for name, value in seeks.items()}, "oracle_seek": min(seeks.values()),
                    "oracle_labels": "|".join(winners), "tie": len(winners) > 1,
                    "label": winners[0] if len(winners) == 1 else None,
                })
    return pd.DataFrame(rows)

