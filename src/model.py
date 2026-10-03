"""Small, interpretable learned scheduler selector."""

from sklearn.tree import DecisionTreeClassifier

from features import FEATURE_COLUMNS


def train_selector(train_frame):
    primary = train_frame.loc[~train_frame["tie"]].copy()
    if primary["label"].nunique() < 2:
        raise RuntimeError("training data has fewer than two non-tied oracle classes")
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=12, class_weight="balanced", random_state=42)
    model.fit(primary[FEATURE_COLUMNS], primary["label"])
    return model, primary

