from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo import parse_queue
from features import extract_features


def test_extended_features_identify_straight_sequence():
    features = extract_features([10, 12, 14, 16, 18], head=50)
    assert features["trend_correlation"] == pytest.approx(1.0)
    assert features["fraction_small_gaps"] == 1.0
    assert features["median_gap"] == 2.0
    assert features["max_gap"] == 2.0


def test_demo_accepts_commas_and_spaces():
    assert parse_queue(["10,20", "30"]) == [10, 20, 30]


def test_demo_rejects_invalid_tracks():
    with pytest.raises(ValueError, match="tracks must be"):
        parse_queue(["10", "300"])

