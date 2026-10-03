"""Classical disk schedulers using the CSE-307 course conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

TRACK_MIN = 0
TRACK_MAX = 199


@dataclass(frozen=True)
class ScheduleResult:
    order: list[int]
    total_seek: int


def _validate(requests: Iterable[int], head: int) -> list[int]:
    queue = [int(request) for request in requests]
    if not TRACK_MIN <= head <= TRACK_MAX:
        raise ValueError(f"head must be in [{TRACK_MIN}, {TRACK_MAX}]")
    if any(not TRACK_MIN <= request <= TRACK_MAX for request in queue):
        raise ValueError(f"all requests must be in [{TRACK_MIN}, {TRACK_MAX}]")
    return queue


def _movement(path: list[int]) -> int:
    return sum(abs(right - left) for left, right in zip(path, path[1:]))


def fcfs(requests: Iterable[int], head: int) -> ScheduleResult:
    queue = _validate(requests, head)
    return ScheduleResult(queue, _movement([head, *queue]))


def sstf(requests: Iterable[int], head: int) -> ScheduleResult:
    pending = _validate(requests, head)
    order: list[int] = []
    current = head
    while pending:
        index = min(range(len(pending)), key=lambda i: (abs(pending[i] - current), pending[i]))
        current = pending.pop(index)
        order.append(current)
    return ScheduleResult(order, _movement([head, *order]))


def scan(requests: Iterable[int], head: int) -> ScheduleResult:
    """SCAN toward 0 first, reaching physical track 0 before reversing."""
    queue = _validate(requests, head)
    lower = sorted((r for r in queue if r <= head), reverse=True)
    upper = sorted(r for r in queue if r > head)
    order = [*lower, *upper]
    path = [head, *lower]
    if path[-1] != TRACK_MIN:
        path.append(TRACK_MIN)
    path.extend(upper)
    return ScheduleResult(order, _movement(path))


def cscan(requests: Iterable[int], head: int) -> ScheduleResult:
    """C-SCAN toward 0, then wrap 0 -> 199 with the wrap counted."""
    queue = _validate(requests, head)
    lower = sorted((r for r in queue if r <= head), reverse=True)
    upper = sorted((r for r in queue if r > head), reverse=True)
    order = [*lower, *upper]
    path = [head, *lower]
    if path[-1] != TRACK_MIN:
        path.append(TRACK_MIN)
    path.append(TRACK_MAX)
    path.extend(upper)
    return ScheduleResult(order, _movement(path))


SCHEDULERS: dict[str, Callable[[Iterable[int], int], ScheduleResult]] = {
    "FCFS": fcfs,
    "SSTF": sstf,
    "SCAN": scan,
    "C-SCAN": cscan,
}


def compare_schedulers(requests: Iterable[int], head: int) -> dict[str, int]:
    queue = list(requests)
    return {name: scheduler(queue, head).total_seek for name, scheduler in SCHEDULERS.items()}

