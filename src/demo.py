"""Interactive demonstration of prediction, confidence, and true seek costs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_dataset  # noqa: E402
from features import FEATURE_COLUMNS, extract_features  # noqa: E402
from model import train_selector  # noqa: E402
from schedulers import SCHEDULERS, TRACK_MAX, TRACK_MIN, compare_schedulers  # noqa: E402

TRAIN_HEAD = 50
TRAIN_WINDOW = 50


def load_model():
    csv_path = ROOT / "results" / "train_windows.csv"
    train = pd.read_csv(csv_path) if csv_path.exists() else build_dataset(range(1, 21), split="train")
    model, _ = train_selector(train)
    return model


def parse_queue(tokens: list[str]) -> list[int]:
    text = " ".join(tokens).replace(",", " ")
    try:
        queue = [int(token) for token in text.split()]
    except ValueError as exc:
        raise ValueError("requests must be whole numbers") from exc
    if not queue:
        raise ValueError("enter at least one request")
    invalid = [request for request in queue if not TRACK_MIN <= request <= TRACK_MAX]
    if invalid:
        raise ValueError(f"tracks must be in [{TRACK_MIN}, {TRACK_MAX}]; got {invalid}")
    return queue


def safe_parse(tokens: list[str]) -> list[int]:
    try:
        return parse_queue(tokens)
    except ValueError as exc:
        raise SystemExit(f"Invalid input: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="*", help="track numbers, separated by spaces or commas")
    parser.add_argument("--head", type=int, default=TRAIN_HEAD, help="initial head position")
    parser.add_argument("--random", type=int, metavar="N", help="generate N uniformly random requests")
    parser.add_argument("--seed", type=int, default=0, help="seed for --random")
    args = parser.parse_args()

    if args.random is not None:
        if args.random < 1:
            raise SystemExit("--random must be at least 1")
        queue = np.random.default_rng(args.seed).integers(TRACK_MIN, TRACK_MAX + 1, args.random).tolist()
    elif args.requests:
        queue = safe_parse(args.requests)
    else:
        queue = safe_parse([input(f"Enter track numbers ({TRACK_MIN}-{TRACK_MAX}), separated by spaces or commas: ")])
    if not TRACK_MIN <= args.head <= TRACK_MAX:
        raise SystemExit(f"head must be in [{TRACK_MIN}, {TRACK_MAX}]")

    if len(queue) != TRAIN_WINDOW:
        print(f"Note: the selector was trained on {TRAIN_WINDOW}-request windows; this {len(queue)}-request prediction is less reliable.")
    if args.head != TRAIN_HEAD:
        print(f"Note: the selector was trained with head={TRAIN_HEAD}; head={args.head} is outside its training setup.")

    model = load_model()
    feature_frame = pd.DataFrame([extract_features(queue, args.head)])[FEATURE_COLUMNS]
    probabilities = dict(zip(model.classes_, model.predict_proba(feature_frame)[0]))
    prediction = max(probabilities, key=probabilities.get)
    costs = compare_schedulers(queue, args.head)
    oracle_seek = min(costs.values())
    winners = [name for name, cost in costs.items() if cost == oracle_seek]

    preview = f"{queue[:20]}{' ...' if len(queue) > 20 else ''}"
    print(f"\nRequests ({len(queue)}): {preview}   head={args.head}")
    print("\nTrue total seek:")
    for name in SCHEDULERS:
        marker = "  <- best" if name in winners else ""
        print(f"  {name:7s} {costs[name]:6d}{marker}")
    print(f"\nSelector prediction: {prediction}   confidence: {probabilities[prediction]:.2f}")
    print("  class probabilities: " + ", ".join(f"{name}={value:.2f}" for name, value in probabilities.items()))
    print("  C-SCAN is evaluated but is not a prediction class because it never uniquely won in training.")
    print(f"  selector seek = {costs[prediction]}   oracle seek = {oracle_seek}   regret = {costs[prediction] - oracle_seek}")
    tie_note = f" (tie among {'|'.join(winners)})" if len(winners) > 1 else ""
    print("  verdict: " + ("CORRECT" if prediction in winners else "WRONG") + tie_note)


if __name__ == "__main__":
    main()

