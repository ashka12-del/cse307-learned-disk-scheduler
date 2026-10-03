# CSE-307: Learned Disk-Scheduler Selector

Track 2 term-paper implementation: compare FCFS, SSTF, SCAN, and C-SCAN, then train a small decision tree to select a scheduler from request-window features.

## Conventions

- Physical tracks: 0-199.
- Main experiments: independent, non-overlapping windows of 50 requests; head resets to 50 for every window.
- SCAN moves toward 0 first and reaches physical track 0 before reversing.
- C-SCAN moves toward 0 first, reaches 0, wraps to 199, and counts the full wrap.
- The four schedulers are run offline for each window. Every minimum-seek scheduler is preserved in `oracle_labels`.
- Exact ties remain in the raw CSVs but are excluded from primary classifier training and ordinary classification accuracy. Tie-aware evaluation accepts any oracle winner.
- Training seeds are 1-20, extension-selection seeds are 21-30, and held-out test seeds are 100-110.

The course example `[98, 183, 37, 122, 14, 124, 65, 67]` from head 53 is tested against FCFS=640, SSTF=236, SCAN=236, and C-SCAN=386.

## Workloads and features

The generators produce sequential, uniformly random, and clustered/bursty windows. Two shift timelines are also evaluated: sequential to random and random to bursty. Each phase contains 20 windows for every held-out seed.

The selector sees only eight pre-decision features: request variance, request range, mean absolute consecutive gap, gap coefficient of variation (a burstiness proxy), direction consistency, mean distance from the head, fraction below the head, and fraction above the head. It does not see any scheduler's measured seek cost.

A separately reported random-forest extension adds four sequential-pattern features: absolute track/index correlation, fraction of consecutive gaps at most five, median gap, and maximum gap. It was selected using only validation seeds 21-30; the decision tree remains the main model.

## Setup and reproduction (Windows PowerShell)

For the simplest Windows workflow, double-click **`Run Project.bat`**. On the first run it creates a local `.venv`, installs any missing pinned packages, runs the tests and complete experiment, and then opens the results folder. If Python is not installed, it displays installation instructions instead of closing silently.

For a live custom-queue demonstration, double-click **`Run Demo.bat`**, or run one of these commands after setup:

```powershell
.\.venv\Scripts\python.exe src\demo.py
.\.venv\Scripts\python.exe src\demo.py 98 183 37 122 14 124 65 67 --head 53
.\.venv\Scripts\python.exe src\demo.py --random 50 --seed 7
```

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

The learned selector improves on the **always-SSTF baseline** by 8,702 tracks (6.53%). Its mean regret was 8.00 tracks per window, and it achieved zero regret on 75.61% of windows. Mean confidence was 0.877 for correct versus 0.778 for wrong predictions on the primary non-tied evaluation; 10-bin ECE was 0.148.

### Per-workload results

| Workload | FCFS | SSTF | SCAN | C-SCAN | Oracle | Selector | Tie-aware accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 44,549 | 38,504 | 46,291 | 84,450 | 35,020 | 39,214 | 62.7% |
| Random | 729,441 | 63,558 | 54,054 | 86,664 | 53,398 | 54,098 | 70.9% |
| Bursty | 128,114 | 31,214 | 44,392 | 80,193 | 30,878 | 31,262 | 93.2% |

Sequential windows are the main weakness: their mean regret is 19.06 tracks/window, versus 3.18 for random and 1.75 for bursty windows. The confusion matrix is retained so these errors are visible rather than hidden.

C-SCAN never uniquely wins because the head starts at 50, movement begins toward 0, and the full 0-to-199 wrap is counted. This is a consequence of the stated geometry and seek-only metric, not a claim that C-SCAN is universally inferior.

### Validation-selected extension and sensitivity

The random forest was selected on validation data without examining the test results. On test data it reaches 79.09% tie-aware accuracy, 70.91% sequential accuracy, and ECE 0.051. Its mean regret is 8.90, however, so the simpler tree remains the main seek-minimizing result. Logistic regression is a weaker baseline at 73.33% accuracy and 10.44 mean regret.

Retraining the main tree with window sizes 30, 50, and 100 gives tie-aware accuracies of 81.52%, 75.61%, and 84.39%. SSTF remains the best fixed policy at all three sizes, and the selector remains better than the best fixed-policy baseline. These values describe deterministic synthetic experiments, not real-device performance. `results/summary.json` and the raw CSVs are the source of truth.

## Repository map

```text
src/schedulers.py   classical algorithms and seek accounting
src/workloads.py    synthetic workloads and shift timelines
src/features.py     main and extended request-window features
src/dataset.py      offline oracle labels with explicit ties
src/model.py        main tree plus validation-controlled baselines
src/evaluate.py     metrics, calibration, shifts, and charts
src/analysis_extras.py per-workload, model, and sensitivity analyses
src/demo.py         interactive custom-queue demonstration
src/run_all.py      complete deterministic pipeline
tests/              course-example and edge-case unit tests
results/            raw/generated evidence
report/report.pdf   final term paper (cover plus four numbered pages)
report/report.tex   editable LaTeX source
```

## Limitations

This is a synthetic, seek-distance-only simulation. It does not model rotational latency, caching, SSD behavior, concurrent arrivals during service, request priorities, or starvation. A fixed initial head and fixed SCAN direction simplify comparison but limit external validity. Oracle labels are computed after observing a complete window, whereas the learned model uses summary features and cannot guarantee the oracle choice. Confidence is empirical class probability from a small decision tree, not a formal uncertainty guarantee.

## AI-use disclosure

Generative AI assisted with project scaffolding, Python implementation, documentation, and debugging. The experiment uses deterministic code, fixed seeds, unit tests, checked-in raw results, and reproducible commands so every numerical claim can be independently verified. The student remains responsible for understanding, reviewing, explaining, and defending the algorithms, experiment, analysis, citations, and final paper.

## References

1. A. Silberschatz, P. B. Galvin, and G. Gagne, *Operating System Concepts*, 10th ed., Wiley, 2018.
2. L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*, Wadsworth, 1984.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

