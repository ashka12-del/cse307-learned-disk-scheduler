"""Evaluation, calibration, shift experiments, and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset import label_window
from features import FEATURE_COLUMNS, extract_features
from schedulers import SCHEDULERS
from workloads import shifted_windows


def expected_calibration_error(correct: np.ndarray, confidence: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence <= upper if index == bins - 1 else confidence < upper)
        if not mask.any():
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        count = int(mask.sum())
        ece += (count / len(confidence)) * abs(accuracy - mean_confidence)
        rows.append({"bin_lower": lower, "bin_upper": upper, "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return float(ece), pd.DataFrame(rows)


def evaluate_test(model, test_frame: pd.DataFrame, results_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    evaluation = test_frame.copy()
    probabilities = model.predict_proba(evaluation[FEATURE_COLUMNS])
    evaluation["prediction"] = model.predict(evaluation[FEATURE_COLUMNS])
    evaluation["confidence"] = probabilities.max(axis=1)
    evaluation["correct_any_oracle"] = [prediction in labels.split("|") for prediction, labels in zip(evaluation["prediction"], evaluation["oracle_labels"])]
    evaluation["selector_seek"] = [row[f"seek_{prediction}"] for (_, row), prediction in zip(evaluation.iterrows(), evaluation["prediction"])]
    evaluation["regret"] = evaluation["selector_seek"] - evaluation["oracle_seek"]

    primary = evaluation.loc[~evaluation["tie"]].copy()
    primary["correct"] = primary["prediction"] == primary["label"]
    correct = primary["correct"].to_numpy(dtype=bool)
    confidence = primary["confidence"].to_numpy(dtype=float)
    ece, reliability = expected_calibration_error(correct, confidence)
    reliability.to_csv(results_dir / "calibration.csv", index=False)

    labels = list(SCHEDULERS)
    matrix = confusion_matrix(primary["label"], primary["prediction"], labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(results_dir / "confusion_matrix.csv")

    seek_totals = {name: int(evaluation[f"seek_{name}"].sum()) for name in labels}
    seek_totals.update(oracle=int(evaluation["oracle_seek"].sum()), selector=int(evaluation["selector_seek"].sum()))
    summary = {
        "test_windows_total": int(len(test_frame)),
        "test_windows_primary_non_tied": int(len(primary)),
        "test_ties": int(test_frame["tie"].sum()),
        "accuracy_primary_non_tied": float(accuracy_score(primary["label"], primary["prediction"])),
        "accuracy_all_tie_aware": float(evaluation["correct_any_oracle"].mean()),
        "total_regret": int(evaluation["regret"].sum()),
        "mean_regret": float(evaluation["regret"].mean()),
        "zero_regret_fraction": float((evaluation["regret"] == 0).mean()),
        "mean_confidence_correct_primary": float(primary.loc[primary["correct"], "confidence"].mean()),
        "mean_confidence_wrong_primary": float(primary.loc[~primary["correct"], "confidence"].mean()),
        "ece_10_bins": ece,
        "total_seek": seek_totals,
    }
    evaluation.to_csv(results_dir / "test_predictions.csv", index=False)
    return summary, evaluation


def evaluate_shifts(model, results_dir: Path, head: int = 50) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    for first, second in [("sequential", "random"), ("random", "bursty")]:
        scenario = f"{first}_to_{second}"
        for seed in range(100, 111):
            for time_index, (kind, requests) in enumerate(shifted_windows(first, second, seed)):
                seeks, winners = label_window(requests, head)
                features = pd.DataFrame([extract_features(requests, head)])[FEATURE_COLUMNS]
                prediction = str(model.predict(features)[0])
                confidence = float(model.predict_proba(features).max())
                rows.append({
                    "scenario": scenario, "seed": seed, "time_index": time_index, "phase": kind,
                    "oracle_labels": "|".join(winners), "prediction": prediction, "confidence": confidence,
                    "oracle_seek": min(seeks.values()), "selector_seek": seeks[prediction],
                    "regret": seeks[prediction] - min(seeks.values()),
                    **{f"seek_{name}": value for name, value in seeks.items()},
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "shift_results.csv", index=False)
    summary = {}
    for scenario, group in frame.groupby("scenario"):
        before = group[group["time_index"] < 20]
        after = group[group["time_index"] >= 20]
        summary[scenario] = {
            "mean_regret_before": float(before["regret"].mean()),
            "mean_regret_after": float(after["regret"].mean()),
            "mean_confidence_before": float(before["confidence"].mean()),
            "mean_confidence_after": float(after["confidence"].mean()),
        }
    return summary, frame


def make_figures(evaluation: pd.DataFrame, shift_frame: pd.DataFrame, results_dir: Path) -> None:
    totals = {name: evaluation[f"seek_{name}"].sum() for name in SCHEDULERS}
    totals["Oracle"] = evaluation["oracle_seek"].sum()
    totals["Selector"] = evaluation["selector_seek"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(totals.keys(), totals.values(), color=["#8093f1"] * 4 + ["#2a9d8f", "#e76f51"])
    ax.set_ylabel("Total head movement (tracks)")
    ax.set_title("All independent held-out test windows")
    ax.tick_params(axis="x", rotation=20)
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "total_seek_comparison.png", dpi=180)
    plt.close(fig)

    reliability = pd.read_csv(results_dir / "calibration.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(reliability["mean_confidence"], reliability["accuracy"], "o-", color="#e76f51", label="Selector")
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1), title="Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "calibration_reliability.png", dpi=180)
    plt.close(fig)

    matrix = pd.read_csv(results_dir / "confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(xlabel="Predicted", ylabel="Oracle", title="Confusion matrix")
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    grouped = shift_frame.groupby(["scenario", "time_index"], as_index=False)[["regret", "confidence"]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for scenario, group in grouped.groupby("scenario"):
        label = scenario.replace("_", " ")
        axes[0].plot(group["time_index"], group["regret"], label=label)
        axes[1].plot(group["time_index"], group["confidence"], label=label)
    for ax in axes:
        ax.axvline(19.5, linestyle="--", color="black", linewidth=1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean selector regret")
    axes[1].set_ylabel("Mean confidence")
    axes[1].set_xlabel("Window index")
    axes[0].legend()
    fig.suptitle("Selector behavior across workload shifts")
    fig.tight_layout()
    fig.savefig(results_dir / "shift_behavior.png", dpi=180)
    plt.close(fig)


def write_summary(summary: dict[str, object], results_dir: Path) -> None:
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

