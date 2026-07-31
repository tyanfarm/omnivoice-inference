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
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol

import anyio
import anyio.to_thread
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

from admission import MAX_STREAMS
from audio_formats import AudioEncoder
from batch_scheduler import GenerationJob

logger = logging.getLogger(__name__)

WS_IDLE_TIMEOUT_S = float(os.environ.get("OMNIVOICE_WS_IDLE_TIMEOUT_S", "60.0"))
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


class SpeechEngine(Protocol):
    """What a session needs from the service, and nothing more.

    Declared here rather than imported so this module stays below
    streaming_api_omnivoice in the import graph. OmniVoiceStreamingService
    satisfies it structurally; no registration or subclassing is involved.
    """

    scheduler: object  # BatchScheduler: .submit(job), .sampling_rate

    def split_text_for_streaming(
        self, text: str, max_chars: int | None = None
    ) -> list[str]:
        ...

    def chunk_speed(self, text: str, requested: float | None) -> float:
        ...

    def audio_chunk_is_empty(self, audio) -> bool:
        ...

    def audio_to_int16_bytes(self, audio) -> bytes:
        ...


@dataclass
class SessionConfig:
    """Everything a session needs that the socket itself does not carry.

    Only `voice` and `response_format` come from the client. Every other field
    is an internal default: the generation knobs match StreamRequest's, so a
    socket and an HTTP request generate with identical settings, and the
    timeouts read the module constants through default_factory so tests can
    patch them. Exposing no tuning knobs is deliberate — the same choice
    /v1/audio/speech makes.

    `chunk_chars` is the module-level STREAM_TEXT_CHUNK_SIZE default rather
    than StreamRequest's 120, so chunk boundaries can differ from the HTTP
    path's for the same text.
    """

    voice: str = "vf_phuong"
    response_format: str = "pcm"
    chunk_chars: int = 240
    language: str | None = None
    speed: float | None = 1.0
    num_step: int = 16
    denoise: bool = False
    postprocess_output: bool = False
    idle_timeout_s: float = field(default_factory=lambda: WS_IDLE_TIMEOUT_S)
    max_session_s: float = field(default_factory=lambda: WS_MAX_SESSION_S)


class WebSocketSpeechSession:
    """One socket: text in, audio out.

    Two tasks share a bounded stream. The reader turns messages into complete
    sentences; the speaker turns sentences into audio one at a time. Splitting
    them means a long utterance never stops the client from sending more text,
    and the bounded stream applies backpressure when the GPU falls behind.

    Neither task lets an exception reach the task group. anyio wraps anything
    that escapes a group in an ExceptionGroup, so `except WebSocketDisconnect`
    around the group would never match; disconnects and generation failures are
    handled where they happen and end the session by cancelling the group.
    """

    def __init__(
        self,
        socket: WebSocket,
        engine: SpeechEngine,
        config: SessionConfig,
        encoder: AudioEncoder,
        ref_audio: str,
        ref_text: str,
    ) -> None:
        self._socket = socket
        self._engine = engine
        self._config = config
        self._encoder = encoder
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._buffer = SentenceBuffer(
            split=engine.split_text_for_streaming, max_chars=config.chunk_chars
        )
        self._pending: GenerationJob | None = None
        self._closed = False
        # Set the moment a read or write shows the peer is gone, so no later
        # step tries to explain itself to a socket nobody is holding.
        self._client_gone = False
        self._scope: anyio.CancelScope | None = None

    async def run(self) -> None:
        send, receive = anyio.create_memory_object_stream(
            max_buffer_size=PENDING_CHUNKS
        )
        try:
            leading = self._encoder.begin()
            if leading and not await self._send_bytes(leading):
                return

            async with anyio.create_task_group() as tg:
                self._scope = tg.cancel_scope
                tg.start_soon(self._speak, receive)
                async with send:
                    with anyio.move_on_after(self._config.max_session_s):
                        await self._read(send)

            await self._finish()
        finally:
            # The client is gone or done; a job still queued has no listener.
            if self._pending is not None:
                self._pending.cancel()

    # -- reader --------------------------------------------------------

    async def _read(self, send) -> None:
        """Socket -> sentences. Returns when the client is finished or idle."""
        while True:
            try:
                with anyio.fail_after(self._config.idle_timeout_s):
                    message = await self._socket.receive_json()
            except TimeoutError:
                # No text for idle_timeout_s: treat the utterance as finished.
                # Whatever is buffered still gets spoken before the close.
                logger.info(
                    "websocket idle for %.1fs; finishing the session",
                    self._config.idle_timeout_s,
                )
                await self._emit(send, self._buffer.drain())
                return
            except WebSocketDisconnect:
                logger.info("websocket session closed by the client")
                self._abandon()
                return
            except json.JSONDecodeError:
                await self._error("Expected a JSON object per frame", param=None)
                return

            # A frame carrying valid JSON that is not an object falls through
            # to the unknown-type error rather than crashing the handler.
            kind = message.get("type") if isinstance(message, dict) else None
            if kind == "text":
                await self._emit(
                    send, self._buffer.append(str(message.get("text", "")))
                )
                if self._buffer.pending_chars > MAX_PENDING_CHARS:
                    await self._error(
                        f"buffered text exceeded {MAX_PENDING_CHARS} characters "
                        "with no sentence end",
                        param="text",
                    )
                    return
            elif kind == "flush":
                await self._emit(send, self._buffer.drain())
            elif kind == "done":
                await self._emit(send, self._buffer.drain())
                return
            else:
                await self._error(
                    f"unknown message type '{kind}'. Expected text, flush, or done.",
                    param="type",
                )
                return

    async def _emit(self, send, chunks: list[str]) -> None:
        for chunk in chunks:
            # Blocks once PENDING_CHUNKS are queued, which stops us reading the
            # socket and pushes back on the client over TCP.
            await send.send(chunk)

    # -- speaker -------------------------------------------------------

    async def _speak(self, receive) -> None:
        """Sentences -> audio frames, strictly one job at a time."""
        try:
            async with receive:
                async for chunk in receive:
                    await self._speak_one(chunk)
        except Exception:
            # Anything from the model or the encoder. Left to propagate it
            # would surface as an ExceptionGroup nobody catches, and the client
            # would sit on a socket that never speaks again.
            logger.exception("websocket session failed while generating audio")
            await self._error(
                "Audio generation failed",
                error_type="server_error",
                code=1011,
            )
            self._abandon()

    async def _speak_one(self, chunk: str) -> None:
        job = GenerationJob(
            text=chunk,
            ref_audio=self._ref_audio,
            ref_text=self._ref_text,
            language=self._config.language,
            speed=self._engine.chunk_speed(chunk, self._config.speed),
            num_step=self._config.num_step,
            denoise=self._config.denoise,
            postprocess_output=self._config.postprocess_output,
        )
        self._pending = job
        self._engine.scheduler.submit(job)
        try:
            audio = await await_job(job)
        finally:
            self._pending = None

        if self._engine.audio_chunk_is_empty(audio):
            return
        data = self._encoder.encode(self._engine.audio_to_int16_bytes(audio))
        if data:
            await self._send_bytes(data)

    # -- termination ---------------------------------------------------

    async def _finish(self) -> None:
        if self._closed or self._client_gone:
            return
        trailing = self._encoder.flush()
        if trailing:
            await self._send_bytes(trailing)
        await self._send_json({"type": "done"})
        await self._close(1000)

    async def _error(
        self,
        message: str,
        param: str | None = None,
        error_type: str = "invalid_request_error",
        code: int = 1008,
    ) -> None:
        await self._send_json(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": param,
                    "code": None,
                }
            }
        )
        await self._close(code)

    def _abandon(self) -> None:
        """End the session now: stop both tasks, send nothing more."""
        self._client_gone = True
        if self._scope is not None:
            self._scope.cancel()

    async def _send_json(self, payload: dict) -> bool:
        if self._closed or self._client_gone:
            return False
        try:
            await self._socket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError) as exc:
            self._peer_vanished(exc)
            return False
        return True

    async def _send_bytes(self, data: bytes) -> bool:
        if self._closed or self._client_gone:
            return False
        try:
            await self._socket.send_bytes(data)
        except (WebSocketDisconnect, RuntimeError) as exc:
            self._peer_vanished(exc)
            return False
        return True

    def _peer_vanished(self, exc: BaseException) -> None:
        # The client went away between our last read and this write. There is
        # nobody left to report to, so end the session quietly rather than
        # logging a traceback per dropped connection.
        logger.info("websocket send failed, treating the client as gone: %s", exc)
        self._abandon()

    async def _close(self, code: int) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client_gone:
            return
        try:
            await self._socket.close(code=code)
        except RuntimeError as exc:
            logger.info("websocket already closed: %s", exc)
