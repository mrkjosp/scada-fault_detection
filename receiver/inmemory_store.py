"""
In-memory telemetry store — a rolling buffer of the last N readings per machine.

This replaces the InfluxDB write path. Telemetry is held in a fixed-length
`deque` per machine; when it fills, the oldest reading is dropped automatically.
This is deliberately simple and dependency-free.

Production note: for retention and querying you would swap this for a
time-series database like InfluxDB behind the SAME `add()` / `history()`
interface — nothing upstream (the MQTT receiver, the dashboard) would need
to change.
"""
from __future__ import annotations

from collections import deque
from threading import Lock

from config.settings import MACHINE_NAMES

# Last N readings kept per machine. ~5 min at 1 Hz telemetry.
DEFAULT_MAXLEN = 300


class InMemoryStore:
    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        self._buffers = {m: deque(maxlen=maxlen) for m in MACHINE_NAMES}
        self._lock = Lock()

    def add(self, machine: str, reading: dict) -> None:
        """Append one verified reading for a machine (thread-safe).

        An unrecognized machine name is still stored, but logged -- normally
        unreachable since the receiver subscribes to exact topic strings, but
        kept as a guard in case that ever changes to a wildcard subscription.
        """
        with self._lock:
            if machine not in self._buffers:
                print(f"[InMemoryStore] WARNING: reading for unrecognized "
                      f"machine {machine!r} (not in {MACHINE_NAMES}). "
                      f"Storing it, but check config.settings.TOPICS.")
                self._buffers[machine] = deque(maxlen=DEFAULT_MAXLEN)
            self._buffers[machine].append(reading)

    def history(self, machine: str, limit: int | None = None) -> list:
        """Return readings for a machine, newest last; optional tail limit.

        limit=None means "everything". limit<=0 means "nothing": this must
        be an explicit check, not `buf[-limit:]` -- when limit==0, -0 == 0
        in Python, so buf[-0:] is buf[0:], the whole list, not empty.
        """
        with self._lock:
            buf = list(self._buffers.get(machine, []))
        if limit is None:
            return buf
        if limit <= 0:
            return []
        return buf[-limit:]

    def latest(self, machine: str) -> dict | None:
        with self._lock:
            buf = self._buffers.get(machine)
            return buf[-1] if buf else None

    def snapshot(self) -> dict:
        """Latest reading for each KNOWN machine (MACHINE_NAMES) -- for a
        dashboard overview. Does not include auto-vivified "unrecognized
        machine" buffers from add(), so a topic misconfiguration shows up as
        a logged warning, not a surprise extra card in the UI. Use
        history()/latest() directly to inspect an unrecognized machine.
        """
        return {m: self.latest(m) for m in MACHINE_NAMES}


# A process-wide singleton the receiver writes to and the dashboard reads from.
STORE = InMemoryStore()
