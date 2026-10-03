"""Report-oriented comparisons that do not alter the main decision tree."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset import build_dataset
from evaluate import expected_calibration_error
from features import BASIC_FEATURE_COLUMNS, EXTENDED_FEATURE_COLUMNS
from model import train_forest_extension, train_logistic_baseline, train_selector


def score_model(model, frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    predictions = model.predict(frame[columns])
    probabilities = model.predict_proba(frame[columns])
    confidence = probabilities.max(axis=1)
    winners = [set(value.split("|")) for value in frame["oracle_labels"]]
    correct_any = np.array([prediction in winner for prediction, winner in zip(predictions, winners)])
    selector_seek = np.array([row[f"seek_{prediction}"] for (_, row), prediction in zip(frame.iterrows(), predictions)])
    regret = selector_seek - frame["oracle_seek"].to_numpy()

    non_tied = ~frame["tie"].to_numpy(dtype=bool)
    primary_correct = predictions[non_tied] == frame.loc[~frame["tie"], "label"].to_numpy()
    ece, _ = expected_calibration_error(primary_correct, confidence[non_tied])
    sequential = frame["workload"].to_numpy() == "sequential"
    return {
        "accuracy_tie_aware": float(correct_any.mean()),
        "sequential_accuracy_tie_aware": float(correct_any[sequential].mean()),
        "mean_regret": float(regret.mean()),
        "sequential_mean_regret": float(regret[sequential].mean()),
        "ece_primary_non_tied": float(ece),
        "confidence_correct_primary": float(confidence[non_tied][primary_correct].mean()),
        "confidence_wrong_primary": float(confidence[non_tied][~primary_correct].mean()),
        "selector_total_seek": int(selector_seek.sum()),
    }


def model_comparison(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    tree, _ = train_selector(train)
    logistic = train_logistic_baseline(train)
    forest = train_forest_extension(train)
    models = {
        "decision_tree_main": (tree, BASIC_FEATURE_COLUMNS),
        "logistic_baseline": (logistic, BASIC_FEATURE_COLUMNS),
        "random_forest_extension": (forest, EXTENDED_FEATURE_COLUMNS),
    }
    rows = []
    validation_scores = {}
    for name, (model, columns) in models.items():
        for split, frame in (("validation", validation), ("test", test)):
            scores = score_model(model, frame, columns)
            rows.append({"model": name, "split": split, **scores})
            if split == "validation":
                validation_scores[name] = scores
    selected = max(validation_scores, key=lambda name: (validation_scores[name]["accuracy_tie_aware"], -validation_scores[name]["mean_regret"]))
    result = pd.DataFrame(rows)
    result["selected_on_validation"] = result["model"] == selected
    result.to_csv(results_dir / "model_comparison.csv", index=False)
    return result


def per_workload_table(evaluation: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    for workload, group in evaluation.groupby("workload", sort=False):
        rows.append({
            "workload": workload,
            "windows": len(group),
            "seek_FCFS": int(group["seek_FCFS"].sum()),
            "seek_SSTF": int(group["seek_SSTF"].sum()),
            "seek_SCAN": int(group["seek_SCAN"].sum()),
            "seek_C-SCAN": int(group["seek_C-SCAN"].sum()),
            "seek_oracle": int(group["oracle_seek"].sum()),
            "seek_selector": int(group["selector_seek"].sum()),
            "selector_accuracy_tie_aware": float(group["correct_any_oracle"].mean()),
            "selector_mean_regret": float(group["regret"].mean()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "per_workload_summary.csv", index=False)
    return table


def sensitivity_check(results_dir: Path) -> pd.DataFrame:
    rows = []
    for window_size in (30, 50, 100):
        train = build_dataset(range(1, 21), window_size=window_size, split="train")
        test = build_dataset(range(100, 111), window_size=window_size, split="test")
        model, _ = train_selector(train)
        scores = score_model(model, test, BASIC_FEATURE_COLUMNS)
        fixed_totals = {name: int(test[f"seek_{name}"].sum()) for name in ("FCFS", "SSTF", "SCAN", "C-SCAN")}
        best_fixed = min(fixed_totals, key=fixed_totals.get)
        rows.append({
            "window_size": window_size,
            **scores,
            **{f"total_seek_{name}": value for name, value in fixed_totals.items()},
            "best_fixed_scheduler": best_fixed,
            "best_fixed_total_seek": fixed_totals[best_fixed],
            "oracle_total_seek": int(test["oracle_seek"].sum()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "window_size_sensitivity.csv", index=False)
    return table

