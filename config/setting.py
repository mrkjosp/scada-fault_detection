"""
Config for the SCADA fault-detection project.

Secrets (MQTT username/password) come from environment variables so nothing
sensitive ends up in the repo - copy .env.example to .env and fill in real
values (load_dotenv() below picks it up automatically).

Everything else here (rated currents, ISO thresholds, nameplate info) is just
physical data about the machines.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

MACHINE_NAMES = ["motor", "pump", "compressor"]
CLASSIFIER_NAMES = ["health", "faulttype", "sensorhealth"]

# rated current (A) per datasheet - used to normalize current_A into a feature
I_RATED = {"motor": 15.2, "pump": 28.0, "compressor": 42.0}

# order matters - training and serving both build the feature vector in
# this exact order, so don't reorder without updating both sides (and the
# firmware's compute_features(), which mirrors this same list)
FEATURE_ORDER = [
    "current_A", "bearing_temp_C", "vibration_rms_mm_s", "vibration_kurtosis",
    "rolling_mean_current_30s", "rolling_std_vibration_30s",
    "temp_rate_of_change", "current_rate_of_change",
    "load_pct", "ambient_temp",
    "current_normalized", "temp_margin",
]

# nameplate + HMI display info, plus ISO 10816-3 vibration zones (mm/s RMS)
# A < B < C < D -> good, acceptable, unsatisfactory, unacceptable
MACHINES = {
    "motor": {
        "name": "MTR-001", "full": "Induction Motor", "model": "ABB M2AA 132M-4",
        "kw": 7.5, "v": "400V", "i": 15.2, "rpm": 1440, "eff": "89.5",
        "csv": "training_data/motor_training_data.csv",
        "iso": {"A": 1.8, "B": 2.8, "C": 4.5, "D": 7.1},
    },
    "pump": {
        "name": "PMP-002", "full": "Centrifugal Pump", "model": "Grundfos CR",
        "kw": 15, "v": "400V", "i": 28.0, "rpm": 2900, "eff": "91.2",
        "csv": "training_data/pump_training_data.csv",
        "iso": {"A": 2.3, "B": 3.5, "C": 5.6, "D": 9.0},
    },
    "compressor": {
        "name": "CMP-003", "full": "Screw Compressor", "model": "Atlas Copco GA22",
        "kw": 22, "v": "400V", "i": 42.0, "rpm": 1460, "eff": "92.0",
        "csv": "training_data/compressor_training_data.csv",
        "iso": {"A": 2.8, "B": 4.5, "C": 7.1, "D": 11.2},
    },
}

# MQTT + service endpoints - all from env, never hardcode creds here
MQTT = {
    "host": os.getenv("MQTT_HOST", "localhost"),
    "port": int(os.getenv("MQTT_PORT", "8883")),
    "username": os.getenv("MQTT_USERNAME", ""),
    "password": os.getenv("MQTT_PASSWORD", ""),
    "use_tls": os.getenv("MQTT_USE_TLS", "true").lower() == "true",
    "qos": int(os.getenv("MQTT_QOS", "1")),
}

FLASK_URL = os.getenv("FLASK_URL", "http://localhost:5000")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# topic -> machine name, used by the receiver to route incoming messages
TOPICS = {
    "factory/machine_1/sensors": "motor",
    "factory/machine_2/sensors": "pump",
    "factory/machine_3/sensors": "compressor",
}