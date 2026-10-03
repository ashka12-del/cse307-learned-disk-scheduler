"""Main selector and lightweight comparison models."""

import warnings

from scipy.optimize import OptimizeWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import BASIC_FEATURE_COLUMNS, EXTENDED_FEATURE_COLUMNS


def train_selector(train_frame):
    primary = train_frame.loc[~train_frame["tie"]].copy()
    if primary["label"].nunique() < 2:
        raise RuntimeError("training data has fewer than two non-tied oracle classes")
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=12, class_weight="balanced", random_state=42)
    model.fit(primary[BASIC_FEATURE_COLUMNS], primary["label"])
    return model, primary


def train_logistic_baseline(train_frame):
    primary = train_frame.loc[~train_frame["tie"]].copy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    # SciPy 1.15 may emit a harmless compatibility warning for L-BFGS' legacy
    # display option even though fitting succeeds deterministically.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        model.fit(primary[BASIC_FEATURE_COLUMNS], primary["label"])
    return model


def train_forest_extension(train_frame):
    primary = train_frame.loc[~train_frame["tie"]].copy()
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(primary[EXTENDED_FEATURE_COLUMNS], primary["label"])
    return model

