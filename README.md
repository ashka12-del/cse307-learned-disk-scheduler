# CSE-307: Learned Disk-Scheduler Selector

Track 2 term-paper implementation: compare FCFS, SSTF, SCAN, and C-SCAN, then train a small decision tree to select a scheduler from request-window features.

## Conventions

- Physical tracks: 0-199.
- Main experiments: independent, non-overlapping windows of 50 requests; head resets to 50 for every window.
- SCAN moves toward 0 first and reaches physical track 0 before reversing.
- C-SCAN moves toward 0 first, reaches 0, wraps to 199, and counts the full wrap.
- The four schedulers are run offline for each window. Every minimum-seek scheduler is preserved in `oracle_labels`.
- Exact ties remain in the raw CSVs but are excluded from primary classifier training and accuracy evaluation.
- Training seeds are 1-20; held-out test seeds are 100-110.

The course example `[98, 183, 37, 122, 14, 124, 65, 67]` from head 53 is tested against FCFS=640, SSTF=236, SCAN=236, and C-SCAN=386.

## Workloads and features

The generators produce sequential, uniformly random, and clustered/bursty windows. Two shift timelines are also evaluated: sequential to random and random to bursty. Each phase contains 20 windows for every held-out seed.

The selector sees only eight pre-decision features: request variance, request range, mean absolute consecutive gap, gap coefficient of variation (a burstiness proxy), direction consistency, mean distance from the head, fraction below the head, and fraction above the head. It does not see any scheduler's measured seek cost.

## Setup and reproduction (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe src\run_all.py
```

On Linux/macOS, activate a virtual environment and replace the final two executable paths with `python`.

The run prints the oracle-label distribution before fitting, writes raw training/test/shift CSVs, predictions, calibration bins, confusion matrix, a machine-readable summary, and four figures under `results/`. Re-running with the pinned dependencies and fixed seeds deterministically replaces those generated files.

## Results

The checked-in run generated 1,200 training windows and 660 held-out test windows. There were 70 training ties and 63 test ties. After excluding training ties, the oracle-label distribution was FCFS 63, SCAN 267, and SSTF 800; C-SCAN was evaluated but was never the unique minimum under these workloads and conventions.

On the 597 non-tied test windows, classifier accuracy was 74.37%. Counting a prediction as correct when it matched any oracle winner on tied windows, all-window accuracy was 75.61%. Across all 660 test windows, total seek was:

| Policy | Total seek |
|---|---:|
| FCFS | 902,104 |
| SSTF | 133,276 |
| SCAN | 144,737 |
| C-SCAN | 251,307 |
| Oracle | 119,296 |
| Learned selector | 124,574 |

The selector's mean regret was 8.00 tracks per window, and it achieved zero regret on 75.61% of windows. Mean confidence was 0.877 for correct versus 0.778 for wrong predictions on the primary non-tied evaluation; 10-bin ECE was 0.148. These values describe this deterministic synthetic experiment, not real-device performance. `results/summary.json` and the raw CSVs are the source of truth for every reported number.

## Repository map

```text
src/schedulers.py   classical algorithms and seek accounting
src/workloads.py    synthetic workloads and shift timelines
src/features.py     eight request-window features
src/dataset.py      offline oracle labels with explicit ties
src/model.py        compact decision-tree selector
src/evaluate.py     metrics, calibration, shifts, and charts
src/run_all.py      complete deterministic pipeline
tests/              course-example and edge-case unit tests
results/            raw/generated evidence
report/             paper workspace placeholder
```

## Limitations

This is a synthetic, seek-distance-only simulation. It does not model rotational latency, caching, SSD behavior, concurrent arrivals during service, request priorities, or starvation. A fixed initial head and fixed SCAN direction simplify comparison but limit external validity. Oracle labels are computed after observing a complete window, whereas the learned model uses summary features and cannot guarantee the oracle choice. Confidence is empirical class probability from a small decision tree, not a formal uncertainty guarantee.

## AI-use disclosure

Generative AI assisted with project scaffolding, Python implementation, documentation, and debugging. The experiment uses deterministic code, fixed seeds, unit tests, checked-in raw results, and reproducible commands so every numerical claim can be independently verified. The student remains responsible for understanding, reviewing, explaining, and defending the algorithms, experiment, analysis, citations, and final paper.

## References

1. A. Silberschatz, P. B. Galvin, and G. Gagne, *Operating System Concepts*, 10th ed., Wiley, 2018.
2. L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*, Wadsworth, 1984.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

