"""
Train the 9 RandomForest classifiers (3 machines x 3 tasks) and emit honest
evaluation artifacts: a classification report, per-classifier confusion matrices,
and feature-importance plots.

Run:
    python -m training.train_models --data-dir training_data --out-dir models

The training data is physics-generated synthetic data. A RandomForest with
max_depth=10 will score near-perfectly on it almost by construction. The metrics
below are valid for "can the model separate the simulated classes", NOT a claim
about real-world field accuracy. Real deployment needs recalibration on measured
data (a domain-gap problem). 
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from config.settings import MACHINE_NAMES, FEATURE_ORDER
from training.feature_engineering import engineer_features

# Map a short classifier name to the dataframe column it predicts.
CLASSIFIERS = {
    "health": "machine_state",
    "faulttype": "fault_type",
    "sensorhealth": "sensor_health",
}


def load_data(data_dir: str) -> pd.DataFrame:
    frames = []
    for machine in MACHINE_NAMES:
        path = os.path.join(data_dir, f"{machine}_training_data.csv")
        mdf = pd.read_csv(path)
        mdf["machine"] = machine
        frames.append(mdf)
        print(f"  loaded {machine}: {mdf.shape}")
    df = pd.concat(frames, ignore_index=True)
    return engineer_features(df)


def sanitize(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the FEATURE columns only: replace inf/NaN, clip to float32 range,
    so the model never sees junk. Deliberately does NOT touch label columns
    or chunk_id.
    """
    label_cols = ["machine_state", "fault_type", "sensor_health", "chunk_id"]
    bad = frame[label_cols].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} row(s) have a NaN label or chunk_id -- fix "
            f"the generator; don't silently coerce to 0. First offending "
            f"row indices: {frame.index[bad].tolist()[:10]}"
        )
    frame = frame.copy()
    f32max = np.finfo(np.float32).max
    frame[FEATURE_ORDER] = (
        frame[FEATURE_ORDER]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=-f32max, upper=f32max)
    )
    return frame


def train_one(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def main(data_dir: str, out_dir: str, make_plots: bool):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading and engineering features...")
    df = load_data(data_dir)

    metrics_summary = {}

    for machine in MACHINE_NAMES:
        mdf = sanitize(df[df["machine"] == machine].copy())
        X = mdf[FEATURE_ORDER].values
        groups = mdf["chunk_id"].values

        for clf_name, target_col in CLASSIFIERS.items():
            y = mdf[target_col].values
            key = f"{machine}_{clf_name}"

            # GROUPED split, not a random row split: rolling_mean_current_30s /
            # rolling_std_vibration_30s are computed with a 30-row rolling
            # window, so consecutive rows share up to 29/30 of their window ,
            # a plain train_test_split scatters that correlated data across
            # both sides and inflates accuracy. chunk_id marks a contiguous
            # ~100-row simulation segment; splitting by chunk_id keeps every
            # row from one segment on the same side of the boundary.
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf = train_one(X_train, y_train)
            y_pred = clf.predict(X_test)

            joblib.dump(clf, os.path.join(out_dir, f"{key}_model.joblib"))

            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            metrics_summary[key] = {
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "classes": list(clf.classes_),
            }
            print(f"  {key:28s} acc={report['accuracy']:.3f} "
                  f"macro-F1={report['macro avg']['f1-score']:.3f}")

            if make_plots:
                _save_confusion(machine, clf_name, y_test, y_pred, clf.classes_, out_dir)
                _save_importance(key, clf, out_dir)

    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved 9 models + metrics_summary.json to {out_dir}/")
    print("Reminder: these metrics are on SYNTHETIC data — state that explicitly.")


def _save_confusion(machine, clf_name, y_test, y_pred, classes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(f"{machine} / {clf_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cm_{machine}_{clf_name}.png"), dpi=110)
    plt.close(fig)


def _save_importance(key, clf, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(clf.feature_importances_)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(FEATURE_ORDER)),
           clf.feature_importances_[order])
    ax.set_xticks(range(len(FEATURE_ORDER)),
                  [FEATURE_ORDER[i] for i in order], rotation=90)
    ax.set_title(f"Feature importance — {key}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fi_{key}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="training_data")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    main(args.data_dir, args.out_dir, make_plots=not args.no_plots)
