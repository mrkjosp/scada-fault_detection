"""
Feature engineering — the SINGLE source of truth shared by training and serving.

Why this matters: the most common production ML bug is computing features one
way during training and a slightly different way during inference. By importing
the same functions in both the training script and the Flask API, the feature
definitions can never drift apart.

The 12 features combine raw sensor readings with derived signals chosen for
physical meaning:
  - rate-of-change features catch *developing* faults (a slow temperature climb)
  - rolling statistics smooth noise and expose trends
  - load/ambient context lets the model separate a real fault from a benign
    change in operating point
"""
import numpy as np
import pandas as pd

from config.settings import I_RATED, FEATURE_ORDER, normalize_machine_id


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived feature columns to a raw sensor dataframe.

    Expects columns: machine, timestamp, current_A, bearing_temp_C,
    vibration_rms_mm_s, vibration_kurtosis, load_pct, ambient_temp.
    Processes each machine independently so rolling windows never span machines.

    Requires a "machine" column: current_normalized needs I_RATED[machine],
    so there's no safe per-row default.
    """
    if "machine" not in df.columns:
        raise ValueError(
            "engineer_features() requires a 'machine' column (one of "
            f"{list(I_RATED.keys())}) , current_normalized depends on "
            "knowing which machine's I_RATED to divide by; there is no "
            "safe default. Set df['machine'] before calling this."
        )

    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values(["machine", "timestamp"]).reset_index(drop=True)

    out_frames = []
    for machine in df["machine"].unique():
        m = df[df["machine"] == machine].copy()

        m["rolling_mean_current_30s"] = (
            m["current_A"].rolling(window=30, min_periods=1).mean()
        )
        m["rolling_std_vibration_30s"] = (
            m["vibration_rms_mm_s"].rolling(window=30, min_periods=1).std().fillna(0)
        )
        m["temp_rate_of_change"] = m["bearing_temp_C"].diff().fillna(0)
        m["current_rate_of_change"] = m["current_A"].diff().fillna(0)

        # I_RATED[normalize_machine_id(...)], with a silent fallback
        rated = I_RATED[normalize_machine_id(machine)]
        m["current_normalized"] = m["current_A"] / rated
        m["temp_margin"] = 80.0 - m["bearing_temp_C"]

        out_frames.append(m)

    return pd.concat(out_frames, ignore_index=True)


def build_feature_vector(data: dict, machine_id: str) -> np.ndarray:
    """Build a single (1, 12) feature row from a sensor payload for inference.

    machine_id may be either the short form ("motor") or the raw
    simulator/edge-node form ("motor_1") , normalize_machine_id() handles
    both, the same way engineer_features() does, so training and serving
    agree on which I_RATED value a given machine gets.

    Derived features are recomputed here only when absent from the payload, so
    an edge node that already computed rolling stats can pass them through, while
    a bare sensor reading still produces a valid vector.
    """
    current = float(data.get("current_A", 0.0))
    bearing_temp = float(data.get("bearing_temp_C", 0.0))
    rated = I_RATED[normalize_machine_id(machine_id)]

    values = {
        "current_A": current,
        "bearing_temp_C": bearing_temp,
        "vibration_rms_mm_s": float(data.get("vibration_rms_mm_s", 0.0)),
        "vibration_kurtosis": float(data.get("vibration_kurtosis", 3.0)),
        "rolling_mean_current_30s": float(data.get("rolling_mean_current_30s", current)),
        "rolling_std_vibration_30s": float(data.get("rolling_std_vibration_30s", 0.0)),
        "temp_rate_of_change": float(data.get("temp_rate_of_change", 0.0)),
        "current_rate_of_change": float(data.get("current_rate_of_change", 0.0)),
        "load_pct": float(data.get("load_pct", 50.0)),
        "ambient_temp": float(data.get("ambient_temp", 25.0)),
        "current_normalized": current / rated,
        "temp_margin": 80.0 - bearing_temp,
    }

    return np.array([[values[name] for name in FEATURE_ORDER]])