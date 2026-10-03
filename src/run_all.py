"""Run the complete reproducible experiment from repository root."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_dataset  # noqa: E402
from evaluate import evaluate_shifts, evaluate_test, make_figures, write_summary  # noqa: E402
from model import train_selector  # noqa: E402


def distribution(frame: pd.DataFrame) -> dict[str, int]:
    return frame["oracle_labels"].value_counts().sort_index().astype(int).to_dict()


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    train = build_dataset(range(1, 21), split="train")
    test = build_dataset(range(100, 111), split="test")
    train.to_csv(results_dir / "train_windows.csv", index=False)
    test.to_csv(results_dir / "test_windows.csv", index=False)
    print("Training oracle-label distribution (ties preserved):", distribution(train))
    print("Test oracle-label distribution (ties preserved):", distribution(test))
    print(f"Training ties excluded from primary fit: {int(train['tie'].sum())}/{len(train)}")
    model, primary_train = train_selector(train)
    evaluation_summary, evaluated = evaluate_test(model, test, results_dir)
    shift_summary, shifts = evaluate_shifts(model, results_dir)
    make_figures(evaluated, shifts, results_dir)
    summary = {
        "configuration": {"tracks": [0, 199], "head": 50, "window_size": 50, "windows_per_workload_per_seed": 20,
                          "train_seeds": [1, 20], "test_seeds": [100, 110],
                          "model": "DecisionTreeClassifier(max_depth=5, min_samples_leaf=12, class_weight=balanced, random_state=42)"},
        "training": {"windows_total": int(len(train)), "windows_primary_non_tied": int(len(primary_train)),
                     "ties": int(train["tie"].sum()), "oracle_label_distribution_including_ties": distribution(train),
                     "primary_label_distribution": primary_train["label"].value_counts().sort_index().astype(int).to_dict()},
        "testing": evaluation_summary, "shifts": shift_summary,
    }
    write_summary(summary, results_dir)
    print("Primary training-label distribution:", summary["training"]["primary_label_distribution"])
    print(f"Primary non-tied accuracy: {evaluation_summary['accuracy_primary_non_tied']:.3f}")
    print(f"All-window tie-aware accuracy: {evaluation_summary['accuracy_all_tie_aware']:.3f}")
    print(f"Mean selector regret: {evaluation_summary['mean_regret']:.3f} tracks/window")
    print(f"ECE (10 bins): {evaluation_summary['ece_10_bins']:.3f}")
    print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()

