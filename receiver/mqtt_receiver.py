"""
MQTT telemetry receiver — verifies CRC32, then writes to the in-memory store.

This replaces the old MQTT->InfluxDB bridge. The CRC32 verification (proving
transport integrity end-to-end with the firmware) lives HERE now, in the
MQTT message handler, instead of in a database bridge.

Flow:  edge node --MQTT--> verify_crc32() --> InMemoryStore --> dashboard

All credentials come from environment variables (see config/settings.py and
.env.example). Nothing sensitive is hard-coded.

Run:
    python -m receiver.mqtt_receiver
"""
import json
import os
import ssl
import time
import uuid
import zlib
from datetime import datetime, timezone
from threading import Lock

import paho.mqtt.client as mqtt

from config.settings import MQTT, TOPICS
from receiver.inmemory_store import STORE

stats = {"received": 0, "valid": 0, "corrupted": 0, "unmapped_topic": 0,
         "per_machine": {"motor": 0, "pump": 0, "compressor": 0}}
stats_lock = Lock()

# Reconnect backoff for the INITIAL connect() only (loop_start() already
# retries a connection that drops mid-session; connect() itself doesn't).
INITIAL_CONNECT_MAX_RETRIES = 5
INITIAL_CONNECT_BACKOFF_S = 3


def verify_crc32(payload: dict) -> bool:
    """Recompute CRC32 over the payload (minus its crc32 field) and compare.

    CRC32 detects (does not correct) transmission bit errors. We serialize
    with sorted keys and no whitespace so the publisher and receiver hash
    identical bytes -- this mirrors crc32_compute()/serialize_verdict() in
    edge_firmware/src/wire_format.c.

    Fields go over the wire as integers (confidence as confidence_pct,
    0-100, not a float) specifically so there's no C-vs-Python
    float-formatting mismatch (trailing zeros, precision) to worry about.
    See tests/test_pipeline.py::test_crc32_matches_real_c_firmware, which
    runs the real C wire_format.c and feeds its output through this function.
    """
    data = payload.copy()
    received = data.pop("crc32", None)
    if received is None:
        return False
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (zlib.crc32(raw) & 0xFFFFFFFF) == received


class Receiver:
    def __init__(self, store=STORE):
        self.store = store
        # Unique per process: a fixed client_id would get a second instance
        # (blue/green deploy, horizontal scaling) disconnected by the
        # broker's duplicate-client-id kick.
        client_id = f"telemetry_receiver_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        # callback_api_version set explicitly -- paho-mqtt 2.x otherwise
        # defaults to the deprecated VERSION1 callback shape (warns on every
        # startup). VERSION2 changes on_connect's signature, see below.
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5,
                                  callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(MQTT["username"], MQTT["password"])
        if MQTT["use_tls"]:
            self.client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
            self.client.tls_insecure_set(False)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None):
        # VERSION2 signature: reason_code is a ReasonCode object, not a bare
        # int -- compares equal to 0 for success, but don't int()-cast or
        # format it assuming it's a plain int.
        if reason_code == 0:
            print("Connected to MQTT broker")
            for topic in TOPICS:
                client.subscribe(topic, qos=MQTT["qos"])
                print(f"  subscribed: {topic}")
        else:
            print(f"MQTT connection failed: {reason_code}")

    def _on_message(self, client, userdata, msg):
        with stats_lock:
            stats["received"] += 1
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            with stats_lock:
                stats["corrupted"] += 1
            return

        if not verify_crc32(payload):
            with stats_lock:
                stats["corrupted"] += 1
            return

        machine = TOPICS.get(msg.topic, "unknown")
        with stats_lock:
            stats["valid"] += 1
            if machine in stats["per_machine"]:
                stats["per_machine"][machine] += 1
            else:
                # Currently unreachable: _on_connect subscribes to exact
                # topic strings from TOPICS, so msg.topic can't be unmapped.
                # Kept defensively in case subscriptions ever move to a
                # wildcard filter (e.g. "factory/+/sensors") without TOPICS
                # being updated to match.
                stats["unmapped_topic"] += 1

        # No de-duplication for QoS>0 redelivery: if MQTT["qos"] is 1 or 2
        # and the broker redelivers, this counts and stores the same reading
        # twice. Would need a monotonic sequence number in the payload (not
        # currently sent by the firmware) to dedupe correctly -- worth
        # knowing before treating stats["valid"] as an exact distinct count.
        payload.pop("crc32", None)
        payload["received_at"] = datetime.now(timezone.utc).isoformat()
        self.store.add(machine, payload)

    def run(self):
        for attempt in range(1, INITIAL_CONNECT_MAX_RETRIES + 1):
            try:
                self.client.connect(MQTT["host"], MQTT["port"], keepalive=60)
                break
            except (OSError, ConnectionRefusedError) as exc:
                if attempt == INITIAL_CONNECT_MAX_RETRIES:
                    raise ConnectionError(
                        f"Could not reach MQTT broker {MQTT['host']}:{MQTT['port']} "
                        f"after {attempt} attempts: {exc}"
                    ) from exc
                print(f"  connect attempt {attempt} failed ({exc}), retrying "
                      f"in {INITIAL_CONNECT_BACKOFF_S}s...")
                time.sleep(INITIAL_CONNECT_BACKOFF_S)

        self.client.loop_start()
        print("Receiver listening. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(10)
                with stats_lock:
                    s = dict(stats)
                print(f"  received={s['received']} valid={s['valid']} "
                      f"corrupted={s['corrupted']} unmapped_topic={s['unmapped_topic']}")
        except KeyboardInterrupt:
            self.client.loop_stop()


if __name__ == "__main__":
    Receiver().run()
