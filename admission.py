from __future__ import annotations

import os
import threading

MAX_STREAMS = int(os.environ.get("OMNIVOICE_MAX_STREAMS", "16"))


class StreamSlot:
    """One admitted stream. Releasing more than once is a no-op."""

    def __init__(self, control: "AdmissionControl") -> None:
        self._control = control
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._control._release()


class AdmissionControl:
    """Caps how many streams may be in flight at once."""

    def __init__(self, max_streams: int = MAX_STREAMS) -> None:
        self._semaphore = threading.BoundedSemaphore(max_streams)
        self._lock = threading.Lock()
        self._active = 0

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def try_acquire(self) -> StreamSlot | None:
        if not self._semaphore.acquire(blocking=False):
            return None
        with self._lock:
            self._active += 1
        return StreamSlot(self)

    def _release(self) -> None:
        with self._lock:
            self._active -= 1
        self._semaphore.release()
