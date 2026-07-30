# WebSocket Streaming TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `WS /v1/audio/speech/ws` so a client can stream text as its LLM produces it and receive audio frames back on the same connection, with the session closing itself ~5s after the text stops.

**Architecture:** One socket is one session. A reader task cuts incoming text into sentences with the existing splitter and pushes chunks onto a bounded in-memory stream; a speaker task pulls them, submits **one job at a time** to the existing `BatchScheduler`, and sends encoded audio back. Because each session submits sequentially, a session holds at most one job in the queue, so the current global FIFO is already fair across sessions and `batch_scheduler.py` needs no changes. Blocking `job.result()` runs on a worker thread so the event loop stays free.

**Tech Stack:** Python 3.10, FastAPI 0.136.3, Starlette 1.1.0, anyio 4.13.0, `websockets` (new runtime dependency, server-side only), existing `BatchScheduler` / `AdmissionControl` / `audio_formats`, pytest.

## Global Constraints

- **Reuse the existing splitter verbatim.** The only new logic is deciding where complete text ends. Streamed chunks are *not* identical to one-shot chunks — the one-shot splitter packs sentences up to `max_chars`, and mid-stream there is no way to know another sentence is coming without stalling until it arrives, which would destroy first-audio latency. What must hold is that every streamed chunk is a boundary the splitter itself would have chosen (`split(chunk) == [chunk]`), no text is lost or reordered, and no chunk ends mid-sentence. Task 2's tests pin exactly that.
- **Imports run one way only.** `ws_session.py` must not import `streaming_api_omnivoice`, which imports it for the route. What the session needs from the service is declared as a local `Protocol` (Task 4), so there is no cycle, no reaching into `_private` names, and no import hidden inside a method body.
- **Sequential per session.** Exactly one `GenerationJob` in flight per socket. Never parallelise a session's chunks — it breaks audio order and refills the queue unfairly.
- **Never block the event loop.** `job.result()` is a blocking `queue.get`; it must run via `anyio.to_thread.run_sync`. One thread is held per *generating* chunk, never for a socket's lifetime.
- **One `AdmissionControl` slot per socket**, from the same `service.admission` as HTTP. Combined cap stays `MAX_STREAMS` (16). Released in a `finally`.
- **Idle disconnect ~5s**, env-tunable via `OMNIVOICE_WS_IDLE_TIMEOUT_S` (default `5.0`). Idle means *no text message*; it flushes the buffer and drains audio before closing, never cuts mid-utterance.
- **Hard session ceiling** via `OMNIVOICE_WS_MAX_SESSION_S` (default `300.0`) so a client trickling one character every 4s cannot hold a slot forever.
- **No changes to `batch_scheduler.py`, `admission.py`, or `audio_formats.py`.**
- **Default `response_format` for WebSocket is `pcm`** (MP3 over a socket needs MSE or per-chunk decode; PCM feeds an AudioWorklet directly). `wav` and `mp3` still work.
- Errors use the same OpenAI envelope as `/v1/audio/speech`, sent as a text frame before the close.
- Close codes: `1000` normal, `1008` bad voice/format, `1011` server-side encoder failure, `1013` over capacity.
- `TestClient.websocket_connect` implements ASGI WebSocket itself — **verified: tests need no `websockets` package**. Only a real uvicorn server does.

## File Structure

| File | Responsibility |
|---|---|
| `streaming_api_omnivoice.py` (modify) | Task 1 promotes the four shared helpers to public names. Task 5 adds the `@app.websocket` route, handshake validation, admission, `_ws_reject`. |
| `ws_session.py` (create) | `SpeechEngine` protocol, `SentenceBuffer`, `await_job`, `SessionConfig`, `WebSocketSpeechSession`. Transport logic only — no FastAPI routing, no import of the app module. |
| `tests/test_sentence_buffer.py` (create) | `SentenceBuffer` in isolation, including parity with the one-shot splitter. |
| `tests/test_await_job.py` (create) | Proves the event loop stays responsive while a chunk generates. |
| `tests/test_ws_speech.py` (create) | Endpoint behaviour through `TestClient`: protocol, ordering, sequencing, idle close, errors, slot accounting. |
| `tests/test_ws_integration.py` (create) | `gpu`-marked: real model, real audio. |
| `requirements.txt`, `README.md` (modify) | Dependency pin and protocol documentation. |

**Resulting import graph** — strictly one direction, no cycles:

```
admission   audio_formats   batch_scheduler   voices      (leaves)
     \            |              /
      \           |             /
            ws_session                                    (session logic)
                  |
       streaming_api_omnivoice                            (routing)
```

---

### Task 1: Promote the helpers both transports share

**Files:**
- Modify: `streaming_api_omnivoice.py:204-241` (`_split_text_for_streaming`), `:296-308` (`_audio_chunk_is_empty`, `_audio_tensor_to_int16_bytes`), `:310-373` (`text_to_speech_stream`)
- Test: the existing suite (`tests/`) must stay green — this is a pure rename plus one extraction.

**Interfaces:**
- Produces, all on `OmniVoiceStreamingService`:
  - `split_text_for_streaming(text: str, max_chars: int | None = None) -> list[str]`
  - `audio_chunk_is_empty(audio) -> bool` (staticmethod)
  - `audio_to_int16_bytes(audio) -> bytes` (classmethod)
  - `chunk_speed(text: str, requested: float | None) -> float` (staticmethod, new)
- `_split_long_text_part`, `_audio_to_numpy`, and `_resolve_ref_audio_path` stay private — nothing outside the class calls them.

Rationale: these four are the entirety of what a non-HTTP transport needs from the service. Making them public is what lets Task 4 depend on a declared interface instead of on `_private` names, and it is why `ws_session` never has to import this module.

- [ ] **Step 1: Rename the three existing helpers**

In `streaming_api_omnivoice.py`:

- `def _split_text_for_streaming(` → `def split_text_for_streaming(`
- `def _audio_chunk_is_empty(` → `def audio_chunk_is_empty(`
- `def _audio_tensor_to_int16_bytes(` → `def audio_to_int16_bytes(`

- [ ] **Step 2: Add the speed rule as a staticmethod**

Insert into `OmniVoiceStreamingService`, next to the other audio helpers:

```python
    @staticmethod
    def chunk_speed(text: str, requested: float | None) -> float:
        """Playback speed for one chunk.

        Very short chunks are read at full speed; slowing a three-word phrase
        makes it sound broken. Public so the WebSocket session speaks text at
        the same rate the HTTP path does.
        """
        speed = requested if requested is not None else 0.8
        if len(text.split()) <= 4:
            return 1.0
        return speed
```

- [ ] **Step 3: Point `text_to_speech_stream` at the new names**

Replace the inline speed block:

```python
            speed = self.chunk_speed(text, request.speed)
```

the splitter call:

```python
            for text_chunk in self.split_text_for_streaming(
                text,
                max_chars=request.chunk_chars,
            ):
```

and the two calls inside the loop:

```python
                if self.audio_chunk_is_empty(audio_chunk):
                    continue

                encoded = encoder.encode(self.audio_to_int16_bytes(audio_chunk))
```

- [ ] **Step 4: Confirm no private references remain**

Run: `grep -n "_split_text_for_streaming\|_audio_chunk_is_empty\|_audio_tensor_to_int16_bytes" *.py tests/*.py bench/*.py`
Expected: no output.

- [ ] **Step 5: Run the whole suite to prove the rename changed nothing**

Run: `python -m pytest -q`
Expected: PASS, `61 passed, 2 deselected` — the same counts as before this task.

- [ ] **Step 6: Commit**

```bash
git add streaming_api_omnivoice.py
git commit -m "refactor: publish the helpers a non-HTTP transport needs"
```

---

### Task 2: SentenceBuffer — incremental sentence cutting

**Files:**
- Create: `ws_session.py`
- Test: `tests/test_sentence_buffer.py`

**Interfaces:**
- Consumes: `OmniVoiceStreamingService.split_text_for_streaming` (Task 1), injected as a callable so this class never imports the app.
- Produces: `SentenceBuffer(split, max_chars, force_after_chars=None)` with `append(text) -> list[str]`, `drain() -> list[str]`, `pending_chars -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sentence_buffer.py
"""SentenceBuffer — deciding which streamed text is ready to speak.

Chunking itself belongs to the service's splitter; this only decides where
complete text ends, so a half-finished sentence is never sent to the model.
"""

from __future__ import annotations

import pytest

from streaming_api_omnivoice import OmniVoiceStreamingService
from ws_session import SentenceBuffer

# An instance, because the splitter is an instance method. Constructing the
# service builds a scheduler but never starts it, so no model is loaded.
SPLIT = OmniVoiceStreamingService().split_text_for_streaming


@pytest.fixture
def buffer():
    return SentenceBuffer(split=SPLIT, max_chars=240)


def test_a_partial_sentence_is_held_back(buffer):
    # Speaking half a clause would ruin prosody and cannot be un-said.
    assert buffer.append("Hello there") == []
    assert buffer.pending_chars == len("Hello there")


def test_a_terminator_releases_the_sentence(buffer):
    assert buffer.append("Hello there.") == ["Hello there."]
    assert buffer.pending_chars == 0


def test_the_tail_after_the_last_terminator_stays_buffered(buffer):
    assert buffer.append("One. Two. Three") == ["One. Two."]
    assert buffer.pending_chars == len(" Three")


def test_text_arriving_letter_by_letter_still_forms_one_sentence(buffer):
    emitted = [chunk for ch in "Hi there." for chunk in buffer.append(ch)]
    assert emitted == ["Hi there."]


def test_drain_releases_text_that_has_no_terminator(buffer):
    buffer.append("No terminator here")
    assert buffer.drain() == ["No terminator here"]
    assert buffer.pending_chars == 0


def test_drain_on_an_empty_buffer_emits_nothing(buffer):
    assert buffer.drain() == []


def test_a_closing_quote_stays_with_its_sentence(buffer):
    # Cutting right after the period would start the next chunk with a stray
    # quote, which the model would try to pronounce.
    assert buffer.append('He said "go." Then left') == ['He said "go."']


def test_a_long_clause_with_no_terminator_is_forced_out(buffer):
    # An LLM emitting a 600-character clause with no period must not stall the
    # stream waiting for one.
    emitted = buffer.append("word " * 120)
    assert emitted, "expected a forced emit once the buffer got long"
    assert buffer.pending_chars == 0


def test_a_forced_emit_still_respects_max_chars(buffer):
    emitted = buffer.append("word " * 200)
    assert emitted
    assert all(len(chunk) <= 240 for chunk in emitted)


def test_streamed_text_chunks_the_same_as_one_shot_text():
    # The whole point of injecting the service's splitter: a sentence must not
    # be cut differently just because it arrived in pieces.
    text = (
        "The mission went farther than ever before. It aimed to return to the "
        "moon. The goal was a permanent colony there. Everyone watched."
    )
    streamed = SentenceBuffer(split=SPLIT, max_chars=120)
    emitted: list[str] = []
    for word in text.split(" "):
        emitted += streamed.append(word + " ")
    emitted += streamed.drain()

    assert emitted == SPLIT(text, max_chars=120)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_sentence_buffer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ws_session'`

- [ ] **Step 3: Write `ws_session.py` with just the buffer**

```python
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

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

WS_IDLE_TIMEOUT_S = float(os.environ.get("OMNIVOICE_WS_IDLE_TIMEOUT_S", "5.0"))
WS_MAX_SESSION_S = float(os.environ.get("OMNIVOICE_WS_MAX_SESSION_S", "300.0"))
# Sentences allowed to wait between the reader and the speaker. Full means the
# reader stops pulling from the socket, which is TCP backpressure for free.
PENDING_CHUNKS = 8
# A client that streams text far faster than the GPU speaks it is a bug, not a
# workload; refuse rather than buffer without bound.
MAX_PENDING_CHARS = 8000


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_sentence_buffer.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add ws_session.py tests/test_sentence_buffer.py
git commit -m "feat: incremental sentence buffer over the existing splitter"
```

---

### Task 3: Await a job without blocking the event loop

**Files:**
- Modify: `ws_session.py`
- Test: `tests/test_await_job.py`

**Interfaces:**
- Consumes: `GenerationJob.result(timeout)` from `batch_scheduler`, `MAX_STREAMS` from `admission`.
- Produces: `await_job(job, timeout=JOB_TIMEOUT_S) -> np.ndarray`, `TTS_THREAD_LIMITER: anyio.CapacityLimiter`, `JOB_TIMEOUT_S: float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_await_job.py
"""await_job must wait on the GPU without freezing the event loop.

If it blocked, one generating chunk would stall every other socket on the
server, which is the whole failure mode WebSocket support has to avoid.
"""

from __future__ import annotations

import threading
import time

import anyio
import numpy as np
import pytest

from batch_scheduler import GenerationJob
from ws_session import await_job


def make_job(text="job-0"):
    return GenerationJob(
        text=text,
        ref_audio="voices/af_heart.mp3",
        ref_text="reference transcript",
        language=None,
        speed=1.0,
        num_step=16,
        denoise=False,
        postprocess_output=False,
    )


def answer_later(job, delay, value):
    def worker():
        time.sleep(delay)
        job.set_result(np.full(4, value, dtype=np.float32))

    threading.Thread(target=worker, daemon=True).start()


def test_await_job_returns_the_audio():
    async def main():
        job = make_job()
        answer_later(job, 0.05, 3.0)
        return await await_job(job, timeout=5)

    assert anyio.run(main).tolist() == [3.0] * 4


def test_await_job_reraises_a_generation_failure():
    async def main():
        job = make_job()
        threading.Thread(
            target=lambda: job.set_exception(RuntimeError("generate blew up")),
            daemon=True,
        ).start()
        await await_job(job, timeout=5)

    with pytest.raises(RuntimeError, match="generate blew up"):
        anyio.run(main)


def test_the_event_loop_keeps_running_while_a_job_generates():
    # The ticker must get to run during the wait. Under a blocking
    # implementation it would only run after the job finished.
    ticks: list[float] = []

    async def ticker():
        for _ in range(5):
            await anyio.sleep(0.02)
            ticks.append(time.monotonic())

    async def main():
        job = make_job()
        answer_later(job, 0.3, 1.0)
        async with anyio.create_task_group() as tg:
            tg.start_soon(ticker)
            await await_job(job, timeout=5)

    anyio.run(main)
    assert len(ticks) == 5, f"event loop was blocked; only {len(ticks)} ticks ran"


def test_two_sessions_wait_concurrently_not_serially():
    # Two sockets each generating a chunk must overlap. Serial waiting would
    # take ~0.6s; concurrent waiting takes ~0.3s.
    async def main():
        jobs = [make_job(), make_job()]
        for job in jobs:
            answer_later(job, 0.3, 1.0)
        started = time.monotonic()
        async with anyio.create_task_group() as tg:
            for job in jobs:
                tg.start_soon(await_job, job, 5)
        return time.monotonic() - started

    assert anyio.run(main) < 0.55
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_await_job.py -q`
Expected: FAIL with `ImportError: cannot import name 'await_job' from 'ws_session'`

- [ ] **Step 3: Add `await_job` to `ws_session.py`**

Extend the imports at the top of `ws_session.py`:

```python
import functools

import anyio
import anyio.to_thread
import numpy as np

from admission import MAX_STREAMS
from batch_scheduler import GenerationJob
```

Add below the module constants:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_await_job.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ws_session.py tests/test_await_job.py
git commit -m "feat: await generation off the event loop with a dedicated limiter"
```

---

### Task 4: The session — reader, speaker, and ordered audio

**Files:**
- Modify: `ws_session.py`, `tests/conftest.py` (`FakeOmniVoice.generate`)
- Test: `tests/test_ws_speech.py`

**Interfaces:**
- Consumes: `SentenceBuffer` (Task 2), `await_job` (Task 3), `AudioEncoder` from `audio_formats`, and the four public service methods from Task 1 — via the `SpeechEngine` protocol declared here, never by importing the app.
- Produces:
  - `SpeechEngine` protocol
  - `SessionConfig(voice="vf_phuong", response_format="pcm", ...)` — only those two are ever passed in; `chunk_chars=240`, `language=None`, `speed=1.0`, `num_step=16`, `denoise=False`, `postprocess_output=False`, `idle_timeout_s`, `max_session_s` are internal defaults.
  - `WebSocketSpeechSession(socket, engine, config, encoder, ref_audio, ref_text)` with `async def run() -> None`

- [ ] **Step 1: Teach FakeOmniVoice to read a job id out of prose**

`FakeOmniVoice.generate` currently does `float(t.split("-")[1])`, which only
survives a text that is a bare `job-<n>` token. Streamed text goes through the
splitter, so one chunk holds several sentences — `"job-0. job-1."` makes that
line raise `ValueError: could not convert string to float: '0. job'`. Verified
against the current harness before writing this plan.

In `tests/conftest.py`, add `import re` and replace the return line of
`generate` plus add a helper:

```python
    def generate(self, text, **kwargs):
        self.calls.append({"text": list(text), **kwargs})
        failing = [t for t in text if t in self.fail_texts]
        if failing:
            raise RuntimeError(f"generate failed for {failing}")
        return [np.full(4, self._job_id(t), dtype=np.float32) for t in text]

    @staticmethod
    def _job_id(text: str) -> float:
        """First job-<n> token in the text.

        Streamed text arrives already packed into chunks of several sentences,
        so the id has to be found rather than assumed to be the whole string.
        """
        match = re.search(r"job-(\d+)", text)
        if match is None:
            raise AssertionError(
                f"FakeOmniVoice needs a 'job-<n>' token in the text, got {text!r}"
            )
        return float(match.group(1))
```

Update the class docstring to say the token may appear anywhere in the text.

Run: `python -m pytest -q`
Expected: PASS, unchanged counts — single-token tests are unaffected.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ws_speech.py
"""WS /v1/audio/speech/ws — the streaming session.

Uses the `client` fixture's FakeOmniVoice, so no GPU. FakeOmniVoice needs a
"job-<n>" token in each text, which doubles as a way to assert ordering.
"""

from __future__ import annotations

import json
import time

import pytest

from audio_formats import WAV_HEADER_SIZE, wav_header


def connect(client, **params):
    query = "&".join(f"{k}={v}" for k, v in {"voice": "af_heart", **params}.items())
    return client.websocket_connect(f"/v1/audio/speech/ws?{query}")


def audio_frames(socket):
    """Collect binary frames up to the terminating JSON frame."""
    frames = []
    while True:
        message = socket.receive()
        if message.get("bytes") is not None:
            frames.append(message["bytes"])
            continue
        return frames, json.loads(message["text"])


def first_json(socket):
    return json.loads(socket.receive()["text"])


def test_a_finished_sentence_produces_audio(client):
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)
    assert frames and all(frames)
    assert final["type"] == "done"


def test_flush_releases_a_sentence_that_has_no_terminator(client):
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1 with no terminator yet"})
        socket.send_json({"type": "flush"})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)
    assert frames, "flush should have released the partial sentence"
    assert final["type"] == "done"


def test_sentences_are_spoken_in_the_order_they_arrived(client):
    client.fake_model.calls.clear()
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-0. job-1. job-2."})
        socket.send_json({"type": "done"})
        audio_frames(socket)

    spoken = [text for call in client.fake_model.calls for text in call["text"]]
    order = [t for t in spoken if t.startswith("job-")]
    assert order == sorted(order), f"audio would play out of order: {order}"


def test_a_session_never_has_two_chunks_generating_at_once(client):
    # Sequential submission is what keeps one session's backlog from crowding
    # the shared queue, so no generate() call may hold two of its chunks. The
    # text has to exceed chunk_chars (240) to be split at all, since the
    # endpoint exposes no chunk_chars knob.
    client.fake_model.calls.clear()
    long_text = " ".join(f"job-{i} " + "filler words here " * 4 + "." for i in range(6))
    assert len(long_text) > 240, "text must be long enough to split"

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": long_text})
        socket.send_json({"type": "done"})
        audio_frames(socket)

    assert len(client.fake_model.calls) > 1, "text should have split into chunks"
    for call in client.fake_model.calls:
        assert len(call["text"]) == 1, f"session chunks were batched together: {call}"


def test_pcm_is_the_default_format(client):
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, _ = audio_frames(socket)
    assert not frames[0].startswith(b"RIFF")


def test_wav_sends_its_header_as_the_first_frame(client):
    with connect(client, response_format="wav") as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, _ = audio_frames(socket)
    assert frames[0][:WAV_HEADER_SIZE] == wav_header(client.fake_model.sampling_rate)


def test_the_voice_from_the_query_string_is_the_one_used(client):
    client.fake_model.prompt_calls.clear()
    with connect(client, voice="am_michael") as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        audio_frames(socket)
    refs = {call[0] for call in client.fake_model.prompt_calls}
    assert refs == {str(client.voices_dir / "am_michael.mp3")}, refs


def test_generation_uses_the_same_defaults_as_the_http_endpoint(client):
    from streaming_api_omnivoice import StreamRequest

    defaults = StreamRequest(text="job-1")
    client.fake_model.calls.clear()
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        audio_frames(socket)

    call = client.fake_model.calls[0]
    assert call["num_step"] == defaults.num_step
    assert call["denoise"] == defaults.denoise
    assert call["postprocess_output"] == defaults.postprocess_output


def test_an_unknown_message_type_is_reported_not_ignored(client):
    with connect(client) as socket:
        socket.send_json({"type": "sing", "text": "job-1."})
        payload = first_json(socket)
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "type"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ws_speech.py -q`
Expected: FAIL — every test errors on the missing route, since Task 5 adds it.

- [ ] **Step 4: Add the protocol, config, and session to `ws_session.py`**

Extend the imports:

```python
from dataclasses import dataclass, field
from typing import Callable, Protocol

from starlette.websockets import WebSocket, WebSocketDisconnect
```

```python
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
    is an internal default: the generation knobs mirror StreamRequest's so a
    socket and an HTTP request speak the same text identically, and the timeouts
    read the module constants through default_factory so tests can patch them.
    Exposing no tuning knobs is deliberate — the same choice /v1/audio/speech
    makes.
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
    """

    def __init__(
        self,
        socket: WebSocket,
        engine: SpeechEngine,
        config: SessionConfig,
        encoder,
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

    async def run(self) -> None:
        send, receive = anyio.create_memory_object_stream(
            max_buffer_size=PENDING_CHUNKS
        )
        try:
            leading = self._encoder.begin()
            if leading:
                await self._socket.send_bytes(leading)

            async with anyio.create_task_group() as tg:
                tg.start_soon(self._speak, receive)
                async with send:
                    with anyio.move_on_after(self._config.max_session_s):
                        await self._read(send)

            await self._finish()
        except WebSocketDisconnect:
            logger.info("websocket session closed by the client")
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

            kind = message.get("type")
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
        async with receive:
            async for chunk in receive:
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
                    continue
                data = self._encoder.encode(self._engine.audio_to_int16_bytes(audio))
                if data:
                    await self._socket.send_bytes(data)

    # -- termination ---------------------------------------------------

    async def _finish(self) -> None:
        trailing = self._encoder.flush()
        if trailing:
            await self._socket.send_bytes(trailing)
        await self._send_json({"type": "done"})
        await self._close(1000)

    async def _error(self, message: str, param: str | None = None) -> None:
        await self._send_json(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": None,
                }
            }
        )
        await self._close(1008)

    async def _send_json(self, payload: dict) -> None:
        if not self._closed:
            await self._socket.send_json(payload)

    async def _close(self, code: int) -> None:
        if self._closed:
            return
        self._closed = True
        await self._socket.close(code=code)
```

Note on `_error` inside `_read`: it closes the socket, so the reader returns and `run()` reaches `_finish()`, which sees `self._closed` already set and sends nothing further.

- [ ] **Step 5: Confirm the import direction holds**

Run: `grep -n "streaming_api_omnivoice" ws_session.py`
Expected: no output. If the name appears anywhere — even inside a function body — the protocol is not being used as intended.

- [ ] **Step 6: Run the tests — they still fail on the missing route**

Run: `python -m pytest tests/test_ws_speech.py -q`
Expected: FAIL, still no route. Task 5 wires it.

- [ ] **Step 7: Commit**

```bash
git add ws_session.py tests/conftest.py
git commit -m "feat: websocket speech session behind a SpeechEngine protocol"
```

---

### Task 5: The endpoint — handshake, admission, close codes

**Files:**
- Modify: `streaming_api_omnivoice.py` (add the route after `create_speech`)
- Test: `tests/test_ws_speech.py` (Task 4's tests, plus the cases below)

**Interfaces:**
- Consumes: `SessionConfig`, `WebSocketSpeechSession` (Task 4), `create_encoder` / `SUPPORTED_FORMATS` (`audio_formats`), `service.admission`, `service.get_voice_clone_config`.
- Produces: route `WS /v1/audio/speech/ws`.

- [ ] **Step 1: Add the failing admission and validation tests**

Append to `tests/test_ws_speech.py`:

```python
def test_an_unknown_voice_is_reported_before_any_audio(client):
    with connect(client, voice="does-not-exist") as socket:
        payload = first_json(socket)
    assert payload["error"]["param"] == "voice"
    assert payload["error"]["type"] == "invalid_request_error"


def test_an_unsupported_format_is_reported(client):
    with connect(client, response_format="opus") as socket:
        payload = first_json(socket)
    assert payload["error"]["param"] == "response_format"
    for name in ("mp3", "wav", "pcm"):
        assert name in payload["error"]["message"]


def test_over_capacity_is_refused_with_a_server_error(client):
    import streaming_api_omnivoice as api

    held = api.service.admission.try_acquire()
    assert held is not None
    try:
        with connect(client) as socket:
            payload = first_json(socket)
        assert payload["error"]["type"] == "server_error"
    finally:
        held.release()


def test_the_slot_is_returned_when_the_session_ends(client):
    import streaming_api_omnivoice as api

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        audio_frames(socket)
    assert api.service.admission.active == 0


def test_a_refused_handshake_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    with connect(client, voice="nope") as socket:
        first_json(socket)
    assert api.service.admission.active == 0


def test_a_client_disconnecting_mid_session_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.receive()  # first audio frame, then walk away
    assert api.service.admission.active == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_ws_speech.py -q`
Expected: FAIL — no route.

- [ ] **Step 3: Add the route**

Extend the imports in `streaming_api_omnivoice.py`:

```python
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket

from ws_session import SessionConfig, WebSocketSpeechSession
```

Add after `create_speech`:

```python
async def _ws_reject(
    socket: WebSocket,
    code: int,
    message: str,
    param: str | None = None,
    error_type: str = "invalid_request_error",
) -> None:
    """Report a handshake failure in the OpenAI envelope, then close.

    The socket is already accepted, so there is no status code left to send —
    a structured frame plus a close code is the only way the client learns why.
    """
    await socket.send_json(
        {"error": {"message": message, "type": error_type, "param": param, "code": None}}
    )
    await socket.close(code=code)


@app.websocket("/v1/audio/speech/ws")
async def speech_websocket(
    socket: WebSocket,
    voice: str = Query(default="vf_phuong", min_length=1),
    response_format: str = Query(default="pcm"),
) -> None:
    """Stream text in, stream audio out, on one connection.

    Not an OpenAI endpoint — OpenAI has no WebSocket TTS — so this protocol is
    ours: JSON text frames in ({"type": "text"|"flush"|"done"}), binary audio
    frames out, one {"type": "done"} at the end. The session closes itself once
    the client has sent no text for OMNIVOICE_WS_IDLE_TIMEOUT_S.

    `voice` and `response_format` are the only knobs. Chunk size, speed, and
    diffusion steps come from SessionConfig's defaults, so a socket and an HTTP
    request produce the same audio for the same text.

    A socket costs one AdmissionControl slot for its lifetime, the same pool
    HTTP streams draw from, so the two together stay under MAX_STREAMS.
    """
    await socket.accept()

    if response_format not in SUPPORTED_FORMATS:
        await _ws_reject(
            socket,
            1008,
            f"Unsupported response_format '{response_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            param="response_format",
        )
        return

    try:
        encoder = create_encoder(response_format, service.scheduler.sampling_rate)
    except RuntimeError as exc:
        await _ws_reject(socket, 1011, str(exc), error_type="server_error")
        return

    try:
        ref_audio_path, ref_text = service.get_voice_clone_config(voice)
    except HTTPException as exc:
        await _ws_reject(socket, 1008, str(exc.detail), param="voice")
        return

    # Taken last, so a rejected handshake can never leak one.
    slot = service.admission.try_acquire()
    if slot is None:
        await _ws_reject(
            socket,
            1013,
            "Too many concurrent streams; retry shortly",
            error_type="server_error",
        )
        return

    session = WebSocketSpeechSession(
        socket=socket,
        engine=service,
        config=SessionConfig(voice=voice, response_format=response_format),
        encoder=encoder,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )
    try:
        await session.run()
    finally:
        slot.release()
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — Task 4's 9 tests and Task 5's 6 now pass alongside the existing 61 and Tasks 2–3's 14.

- [ ] **Step 5: Commit**

```bash
git add streaming_api_omnivoice.py tests/test_ws_speech.py
git commit -m "feat: WS /v1/audio/speech/ws endpoint with admission and close codes"
```

---

### Task 6: Idle disconnect and the hard session ceiling

**Files:**
- Test: `tests/test_ws_speech.py`
- Modify: `ws_session.py` only if a test exposes a gap

**Interfaces:**
- Consumes: `SessionConfig.idle_timeout_s`, `SessionConfig.max_session_s`, module constants `WS_IDLE_TIMEOUT_S` / `WS_MAX_SESSION_S`.

The behaviour was built in Task 4; this task proves it and pins the numbers. Monkeypatching the module constant works because `SessionConfig` reads it through `default_factory` at construction time.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_session_closes_itself_after_the_idle_timeout(client, monkeypatch):
    # The client stops sending text and never says "done" — the common case
    # when an LLM stream just ends. The server must not hold the slot.
    import streaming_api_omnivoice as api
    import ws_session

    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 0.3)

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        frames, final = audio_frames(socket)

    assert frames, "buffered audio should still be delivered"
    assert final["type"] == "done"
    assert api.service.admission.active == 0


def test_idle_flush_speaks_text_that_never_got_a_terminator(client, monkeypatch):
    # A trailing fragment must not be silently dropped when the stream ends.
    import ws_session

    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 0.3)
    client.fake_model.calls.clear()

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-2 without a period"})
        audio_frames(socket)

    spoken = [t for call in client.fake_model.calls for t in call["text"]]
    assert any("job-2" in t for t in spoken), spoken


def test_the_idle_timer_resets_on_every_text_message(client, monkeypatch):
    # A slow LLM sending a word every 0.1s must not be cut off at 0.3s.
    import ws_session

    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 0.3)

    with connect(client) as socket:
        for i in range(5):
            socket.send_json({"type": "text", "text": f"job-{i} "})
            time.sleep(0.1)
        socket.send_json({"type": "text", "text": "end."})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)

    assert final["type"] == "done"
    assert frames


def test_the_hard_ceiling_ends_a_session_that_never_stops(client, monkeypatch):
    # A client trickling text just under the idle timeout forever would hold a
    # slot indefinitely; max_session_s is the backstop.
    import streaming_api_omnivoice as api
    import ws_session

    monkeypatch.setattr(ws_session, "WS_MAX_SESSION_S", 0.5)
    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 5.0)

    with connect(client) as socket:
        with pytest.raises(Exception):
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                socket.send_json({"type": "text", "text": "job-1 "})
                time.sleep(0.1)
            raise AssertionError("session was never closed")

    assert api.service.admission.active == 0
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_ws_speech.py -q -k "idle or ceiling"`
Expected: PASS if Task 4 is correct. If the ceiling test hangs, `anyio.move_on_after` is not wrapping `_read` — fix that in `run()` rather than loosening the test.

- [ ] **Step 3: Confirm the defaults are documented in the module**

`ws_session.py`'s docstring must name both env vars with their defaults (`OMNIVOICE_WS_IDLE_TIMEOUT_S=5.0`, `OMNIVOICE_WS_MAX_SESSION_S=300.0`) and state that idle is measured from the last *text* message.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ws_speech.py ws_session.py
git commit -m "test: idle disconnect and hard session ceiling"
```

---

### Task 7: Dependency, docs, and a GPU test

**Files:**
- Modify: `requirements.txt`, `README.md`
- Create: `tests/test_ws_integration.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Install `websockets` and pin the resolved version**

```bash
uv pip install websockets
python -c "from importlib.metadata import version; print('websockets==' + version('websockets'))"
```

Append the printed pin to `requirements.txt`, with a comment:

```
# WebSocket transport for /v1/audio/speech/ws. uvicorn is pinned without
# [standard], so without this the handshake fails.
websockets==<resolved version>
```

- [ ] **Step 2: Verify a real server actually accepts a WebSocket**

Start the server and connect in the **same** shell invocation — the Bash tool isolates the network per invocation, so a server started earlier is unreachable:

```bash
python -m uvicorn streaming_api_omnivoice:app --port 9000 & SERVER=$!
sleep 45  # model load
python - <<'PY'
import asyncio, json, websockets

async def main():
    url = "ws://127.0.0.1:9000/v1/audio/speech/ws?voice=af_heart&response_format=pcm"
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "text", "text": "Hello there. This is a test."}))
        await ws.send(json.dumps({"type": "done"}))
        total = 0
        async for frame in ws:
            if isinstance(frame, bytes):
                total += len(frame)
            else:
                print("final:", frame)
        print("pcm bytes:", total, "=", total / 2 / 24000, "seconds")

asyncio.run(main())
PY
kill $SERVER
```

Expected: a nonzero byte count and `{"type": "done"}`. A `403` or "Unsupported upgrade request" means Step 1 did not take effect.

- [ ] **Step 3: Write the GPU integration test**

```python
# tests/test_ws_integration.py
"""Real model, real audio over a WebSocket. Run with: pytest -m gpu"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def live_client():
    from fastapi.testclient import TestClient

    import streaming_api_omnivoice as api

    with TestClient(api.app) as client:
        yield client


def collect(socket):
    audio = b""
    while True:
        message = socket.receive()
        if message.get("bytes") is not None:
            audio += message["bytes"]
            continue
        return audio, json.loads(message["text"])


def test_streamed_sentences_return_playable_audio(live_client):
    url = "/v1/audio/speech/ws?voice=af_heart&response_format=wav"
    with live_client.websocket_connect(url) as socket:
        for sentence in ("Hello there. ", "This is a streaming test. ", "Goodbye."):
            socket.send_json({"type": "text", "text": sentence})
        socket.send_json({"type": "done"})
        audio, final = collect(socket)

    assert final["type"] == "done"
    assert audio.startswith(b"RIFF")
    # 24 kHz, 16-bit mono: three sentences must exceed a second of audio.
    assert (len(audio) - 44) / 2 / 24000 > 1.0


def test_two_concurrent_sessions_both_get_audio(live_client):
    url = "/v1/audio/speech/ws?voice=af_heart&response_format=pcm"
    with live_client.websocket_connect(url) as a, live_client.websocket_connect(
        url
    ) as b:
        for socket in (a, b):
            socket.send_json({"type": "text", "text": "Concurrent session test."})
            socket.send_json({"type": "done"})
        for socket in (a, b):
            audio, final = collect(socket)
            assert audio and final["type"] == "done"
```

- [ ] **Step 4: Run it on the GPU box**

Run: `python -m pytest tests/test_ws_integration.py -m gpu -q`
Expected: PASS (2 tests). This loads the real model, so allow ~60s.

- [ ] **Step 5: Document the protocol in `README.md`**

Add after the "OpenAI-compatible endpoint" section:

````markdown
## WebSocket streaming

`WS /v1/audio/speech/ws?voice=vf_phuong&response_format=pcm` streams text in and
audio out on one connection. This is **not** an OpenAI endpoint — OpenAI has no
WebSocket TTS — so the protocol is specific to this server.

Two query parameters, and no others: `voice` (default `vf_phuong`) and
`response_format` (`pcm` default, `wav`, `mp3`). Chunk size, speed, and
diffusion steps use the same defaults as `/v1/audio/speech`.

Client sends JSON text frames:

- `{"type":"text","text":"..."}` — append text as your LLM produces it. Text is
  spoken once a sentence is complete; a trailing fragment waits for more.
- `{"type":"flush"}` — speak the buffer now, terminator or not.
- `{"type":"done"}` — no more text; finish the audio and close.

Server sends binary frames of audio, then one `{"type":"done"}` before closing.
Errors arrive as `{"error":{...}}` in the same envelope the HTTP endpoint uses,
followed by a close code: `1008` bad voice or format, `1013` over capacity,
`1011` server error.

**The session closes itself after 5s with no `text` message**
(`OMNIVOICE_WS_IDLE_TIMEOUT_S`), flushing whatever is buffered first. Set this
above your LLM's worst inter-token gap or a slow response will be cut short.
A hard ceiling of 300s (`OMNIVOICE_WS_MAX_SESSION_S`) ends any session that
never stops.

Each socket holds one of the `MAX_STREAMS` (16) admission slots for its
lifetime — the same pool HTTP streams use, so the two together cannot
oversubscribe the GPU. A session speaks its sentences one at a time, so it can
never have more than one job queued.

```python
import asyncio, json, websockets

async def speak():
    url = "ws://localhost:9000/v1/audio/speech/ws?voice=af_heart&response_format=pcm"
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "text", "text": "Hello there."}))
        await ws.send(json.dumps({"type": "done"}))
        async for frame in ws:
            if isinstance(frame, bytes):
                ...  # 24 kHz 16-bit signed little-endian mono
            else:
                print(json.loads(frame))

asyncio.run(speak())
```
````

- [ ] **Step 6: Run everything and commit**

```bash
python -m pytest -q
git add requirements.txt README.md tests/test_ws_integration.py
git commit -m "docs: websocket protocol, pin websockets, add gpu integration test"
```

---

## Design notes

**Why a protocol instead of extracting a module.** The alternative to Task 4's
`SpeechEngine` is pulling the splitter and audio converters into their own
modules (`text_chunking.py`, and the int16 conversion into `audio_formats.py`)
that both transports import. That is arguably the better long-term shape —
those four helpers are pure functions with no instance state, and they hang off
a service class only for historical reasons. It was not chosen here because it
moves working code for no behavioural gain, and the protocol removes the cycle
just as completely. If a third transport appears, do the extraction then.

## Out of scope

Recorded so they are choices, not oversights:

- **`streaming_player.html`** still uses `GET /api/stream-mp3`. A browser demo of
  the WebSocket path needs an AudioWorklet to play raw PCM — worth its own task.
- **Barge-in / cancel.** No message cancels an utterance mid-flight. A
  `{"type":"cancel"}` that cancels `self._pending` and drains the stream is a
  small addition, but a job already inside `generate()` cannot be stopped, so
  the earliest it can take effect is the next chunk.
- **Fair queuing by `session_id`.** Unnecessary here: sequential per-session
  submission means each session has at most one queued job, so the existing FIFO
  is already fair across sessions.
- **Multi-server routing.** A socket pins a session to one process by
  construction, which is the property that motivated this design. Load balancing
  needs `Upgrade`/`Connection` headers passed through and a `proxy_read_timeout`
  above the session ceiling.
- **Metrics.** No per-session logging or timing beyond the idle/disconnect lines.
  Add a `session_id` to `GenerationJob` if log correlation is wanted later.
