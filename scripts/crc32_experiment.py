"""
CRC32 robustness experiment.

"CRC32-verified telemetry". We generate payloads, inject random single-bit flips at a controlled rate,
and report what fraction of corrupted messages the CRC32 check catches.

Run:
    python -m scripts.crc32_experiment

Expected result: CRC32 catches essentially all single-bit corruptions.
Key point : CRC detects, it does not correct.
"""
import json
import random
import zlib


def make_payload(i: int) -> dict:
    data = {
        "machine_id": "motor",
        "current_A": round(12.0 + random.random(), 3),
        "bearing_temp_C": round(45.0 + random.random() * 5, 3),
        "vibration_rms_mm_s": round(2.0 + random.random(), 3),
        "seq": i,
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data["crc32"] = zlib.crc32(raw) & 0xFFFFFFFF
    return data


def verify(payload: dict) -> bool:
    data = payload.copy()
    received = data.pop("crc32", None)
    if received is None:
        return False
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (zlib.crc32(raw) & 0xFFFFFFFF) == received


def flip_one_bit(blob: bytes) -> bytes:
    b = bytearray(blob)
    idx = random.randrange(len(b))
    bit = 1 << random.randrange(8)
    b[idx] ^= bit
    return bytes(b)


def run(n: int = 20000, corrupt_rate: float = 0.5):
    caught = missed = clean_pass = false_reject = 0
    for i in range(n):
        payload = make_payload(i)
        wire = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        if random.random() < corrupt_rate:
            wire = flip_one_bit(wire)
            try:
                received = json.loads(wire.decode("utf-8"))
                ok = verify(received)
            except (ValueError, UnicodeDecodeError):
                ok = False  # unparseable counts as caught
            if ok:
                missed += 1
            else:
                caught += 1
        else:
            received = json.loads(wire.decode("utf-8"))
            if verify(received):
                clean_pass += 1
            else:
                false_reject += 1

    total_corrupt = caught + missed
    print(f"messages:            {n}")
    print(f"corrupted injected:  {total_corrupt}")
    print(f"  caught:            {caught}")
    print(f"  missed (escaped):  {missed}")
    if total_corrupt:
        print(f"  detection rate:    {caught / total_corrupt * 100:.3f}%")
    print(f"clean messages:      {clean_pass + false_reject}")
    print(f"  passed:            {clean_pass}")
    print(f"  false rejects:     {false_reject}")


if __name__ == "__main__":
    run()
