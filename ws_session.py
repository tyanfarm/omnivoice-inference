"""WebSocket streaming session for /v1/audio/speech/ws.

One socket is one session. A reader task turns incoming text into complete
sentences; a speaker task turns those into audio, one chunk at a time, so a
session never has more than one job waiting on the GPU.

This module must not import streaming_api_omnivoice — that module imports this
one for the route. What the session needs from the service is declared here as
the SpeechEngine protocol and satisfied structurally.

Environment:
    OMNIVOICE_WS_IDLE_TIMEOUT_S   close a session after this long with no text
                                  message (default 5.0)
    OMNIVOICE_WS_MAX_SESSION_S    hard ceiling on a session's life (default 300)
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Callable

import anyio
import anyio.to_thread
import numpy as np

from admission import MAX_STREAMS
from batch_scheduler import GenerationJob

logger = logging.getLogger(__name__)

WS_IDLE_TIMEOUT_S = float(os.environ.get("OMNIVOICE_WS_IDLE_TIMEOUT_S", "5.0"))
WS_MAX_SESSION_S = float(os.environ.get("OMNIVOICE_WS_MAX_SESSION_S", "300.0"))
# Sentences allowed to wait between the reader and the speaker. Full means the
# reader stops pulling from the socket, which is TCP backpressure for free.
PENDING_CHUNKS = 8
# A client that streams text far faster than the GPU speaks it is a bug, not a
# workload; refuse rather than buffer without bound.
MAX_PENDING_CHARS = 8000
# Its own limiter, sized to the admission cap, so TTS waits can never exhaust
# the shared threadpool Starlette needs for its own sync endpoints.
TTS_THREAD_LIMITER = anyio.CapacityLimiter(MAX_STREAMS)
# A chunk that has not come back in this long means the worker died; fail the
# session rather than hold a slot forever.
JOB_TIMEOUT_S = 120.0


async def await_job(job: GenerationJob, timeout: float = JOB_TIMEOUT_S) -> np.ndarray:
    """Wait for one generated chunk without blocking the event loop.

    job.result() is a blocking queue.get, so it runs on a worker thread. The
    thread is held only while that chunk generates, never for the life of the
    socket. abandon_on_cancel lets a disconnect return immediately instead of
    waiting on a get that nobody will answer; the caller cancels the job.
    """
    return await anyio.to_thread.run_sync(
        functools.partial(job.result, timeout),
        limiter=TTS_THREAD_LIMITER,
        abandon_on_cancel=True,
    )


class SentenceBuffer:
    """Accumulates streamed text and hands out only what is ready to speak.

    The tail of a stream is usually half a sentence, so text is released only
    up to the last sentence terminator. Cutting the released text into chunks
    is delegated to the caller's splitter, so streamed text chunks exactly as
    one-shot text does.
    """

    TERMINATORS = ".!?"
    CLOSERS = "\"'"

    def __init__(
        self,
        split: Callable[..., list[str]],
        max_chars: int,
        force_after_chars: int | None = None,
    ) -> None:
        self._split = split
        self._max_chars = max_chars
        # Without this, a long clause carrying no terminator would stall the
        # stream until the client happened to send one.
        self._force_after_chars = force_after_chars or max_chars * 2
        self._buffer = ""

    @property
    def pending_chars(self) -> int:
        return len(self._buffer)

    def append(self, text: str) -> list[str]:
        self._buffer += text
        ready, self._buffer = self._cut()
        return self._chunks(ready)

    def drain(self) -> list[str]:
        """Release everything, terminator or not. For flush, done, and idle."""
        ready, self._buffer = self._buffer.strip(), ""
        return self._chunks(ready)

    def _chunks(self, ready: str) -> list[str]:
        if not ready:
            return []
        return self._split(ready, max_chars=self._max_chars)

    def _cut(self) -> tuple[str, str]:
        cut = max(self._buffer.rfind(t) for t in self.TERMINATORS)
        if cut < 0:
            if len(self._buffer) >= self._force_after_chars:
                return self._buffer.strip(), ""
            return "", self._buffer

        end = cut + 1
        while end < len(self._buffer) and self._buffer[end] in self.CLOSERS:
            end += 1
        return self._buffer[:end].strip(), self._buffer[end:]
