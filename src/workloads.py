"""Deterministic synthetic disk-request workloads."""

from __future__ import annotations

import numpy as np

TRACK_MIN = 0
TRACK_MAX = 199


def _clip(values: np.ndarray) -> list[int]:
    return np.clip(np.rint(values), TRACK_MIN, TRACK_MAX).astype(int).tolist()


def sequential_workload(rng: np.random.Generator, size: int = 50) -> list[int]:
    """Mostly monotonic locality with small timing/order noise."""
    direction = int(rng.choice([-1, 1]))
    step = int(rng.integers(1, 5))
    span = step * (size - 1)
    if direction > 0:
        start = int(rng.integers(TRACK_MIN, max(TRACK_MIN + 1, TRACK_MAX - span + 1)))
    else:
        start = int(rng.integers(min(TRACK_MAX, span), TRACK_MAX + 1))
    values = start + direction * step * np.arange(size) + rng.normal(0, 1.2, size)
    return _clip(values)


def random_workload(rng: np.random.Generator, size: int = 50) -> list[int]:
    return rng.integers(TRACK_MIN, TRACK_MAX + 1, size=size).astype(int).tolist()


def bursty_workload(rng: np.random.Generator, size: int = 50) -> list[int]:
    """Ordered bursts around several separated hot spots."""
    cluster_count = int(rng.integers(2, 5))
    centers = np.sort(rng.choice(np.arange(10, 190), size=cluster_count, replace=False))
    burst_ids: list[int] = []
    while len(burst_ids) < size:
        burst_ids.extend([int(rng.integers(cluster_count))] * int(rng.integers(3, 11)))
    values = np.array([rng.normal(centers[index], rng.uniform(2.0, 7.0)) for index in burst_ids[:size]])
    return _clip(values)


GENERATORS = {
    "sequential": sequential_workload,
    "random": random_workload,
    "bursty": bursty_workload,
}


def generate_workload(kind: str, rng: np.random.Generator, size: int = 50) -> list[int]:
    try:
        return GENERATORS[kind](rng, size)
    except KeyError as exc:
        raise ValueError(f"unknown workload kind: {kind}") from exc


def shifted_windows(first: str, second: str, seed: int, windows_per_phase: int = 20, window_size: int = 50) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(seed)
    timeline = []
    for kind in (first, second):
        for _ in range(windows_per_phase):
            timeline.append((kind, generate_workload(kind, rng, window_size)))
    return timeline

