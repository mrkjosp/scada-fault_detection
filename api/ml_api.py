"""
Flask inference API. Loads the 9 trained models once and serves predictions.

Endpoints:
  GET  /health             - liveness + how many models loaded
  POST /predict            - body: sensor payload incl. machine_id -> verdict
  GET  /history?machine_id=motor&last_n=60 - recent predictions (in-memory)

machine_id accepts either the short form ("motor") or the raw simulator/
edge-node form ("motor_1") -- see config.settings.normalize_machine_id().

Run:
    python -m api.ml_api
"""
import os
import threading
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, jsonify, request

from config.settings import (
    MACHINE_NAMES, CLASSIFIER_NAMES, MODEL_DIR, FEATURE_ORDER,
    normalize_machine_id,
)
from training.feature_engineering import build_feature_vector

EXPECTED_MODEL_COUNT = len(MACHINE_NAMES) * len(CLASSIFIER_NAMES)


def load_models(model_dir: str) -> dict:
    models = {}
    for machine in MACHINE_NAMES:
        models[machine] = {}
        for clf in CLASSIFIER_NAMES:
            path = os.path.join(model_dir, f"{machine}_{clf}_model.joblib")
            try:
                models[machine][clf] = joblib.load(path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Missing model for machine={machine!r} classifier={clf!r} "
                    f"at {path!r}. Run training.train_models first."
                ) from exc
    return models


def create_app(model_dir: str = MODEL_DIR) -> Flask:
    app = Flask(__name__)
    models = load_models(model_dir)
    history = {m: deque(maxlen=500) for m in MACHINE_NAMES}
    history_lock = threading.Lock()

    @app.route("/health", methods=["GET"])
    def health():
        count = sum(len(v) for v in models.values())
        return jsonify({
            "status": "ok",
            "models_loaded": count == EXPECTED_MODEL_COUNT,
            "model_count": count,
            "expected_model_count": EXPECTED_MODEL_COUNT,
            "machines": MACHINE_NAMES,
            "feature_count": len(FEATURE_ORDER),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            # Covers bad JSON (None) and valid-but-non-object JSON (array,
            # number, etc.) -- without this check either case falls through
            # to the generic except below and returns a 500 instead of a 400.
            return jsonify({"error": "request body must be a JSON object"}), 400

        try:
            raw_id = str(data.get("machine_id", ""))
            try:
                machine_id = normalize_machine_id(raw_id.lower())
            except ValueError:
                return jsonify({"error": f"unknown machine_id: {raw_id}"}), 400

            features = build_feature_vector(data, machine_id)
            mm = models[machine_id]

            result = {"machine_id": machine_id,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            for clf_name, out_health, out_conf in [
                ("health", "machine_health", "health_confidence"),
                ("faulttype", "fault_type", "fault_confidence"),
                ("sensorhealth", "sensor_health", "sensor_confidence"),
            ]:
                proba = mm[clf_name].predict_proba(features)[0]
                result[out_health] = mm[clf_name].predict(features)[0]
                result[out_conf] = round(float(np.max(proba)), 4)

            with history_lock:
                history[machine_id].append(result)
            return jsonify(result)
        except Exception as exc:  # noqa: BLE001 - unexpected server-side error
            return jsonify({"error": str(exc)}), 500

    @app.route("/history", methods=["GET"])
    def get_history():
        raw_id = request.args.get("machine_id", "")
        try:
            machine_id = normalize_machine_id(raw_id.lower())
        except ValueError:
            return jsonify({"error": f"unknown machine_id: {raw_id}"}), 400

        last_n_raw = request.args.get("last_n", "60")
        try:
            last_n = int(last_n_raw)
        except ValueError:
            return jsonify({"error": f"last_n must be an integer, got {last_n_raw!r}"}), 400
        if last_n < 0:
            return jsonify({"error": "last_n must be >= 0"}), 400

        with history_lock:
            items = list(history[machine_id])
        return jsonify(items[-last_n:] if last_n > 0 else [])

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, threaded=True)
