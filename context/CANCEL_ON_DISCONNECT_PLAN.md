# Cancel-on-Disconnect Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a WebSocket disconnect actually cancel the session's submitted-but-not-yet-started `GenerationJob`, so a client that closes mid-utterance stops consuming GPU time.

**Architecture:** Two coupled changes. `GenerationJob`'s one-slot result queue becomes first-writer-wins with non-blocking writes, so `cancel()` can release a caller blocked in `result()` without ever deadlocking the scheduler worker. Then `WebSocketSpeechSession._speak_one` stops clearing `self._pending` in a `finally`, so a cancelled `_speak` leaves the job reference intact for `run()`'s `finally` to cancel — exactly how the HTTP path already works.

**Tech Stack:** Python 3.10, `queue.Queue`, `threading.Event`, anyio 4.13.0, FastAPI/Starlette WebSockets, pytest.

## Global Constraints

- **Task 1 must land before Task 2.** Task 2 alone is a regression: it makes `cancel()` fire, the scheduler then skips the job, nobody fills its result slot, and the abandoned worker thread blocks for the full `JOB_TIMEOUT_S` (120.0s) holding one of the 16 `TTS_THREAD_LIMITER` tokens. Sixteen disconnects inside two minutes would exhaust the limiter and stall generation server-wide. Task 1 is what makes Task 2 safe.
- **All three writers to `GenerationJob._slot` must be non-blocking.** `_slot` is `queue.Queue(maxsize=1)`. Changing only `cancel()` to write leaves `set_result`/`set_exception` on a blocking `put()`, so when cancel wins the race the scheduler's single worker thread blocks forever on a full slot and every future request hangs. Verified: a blocking `put` on a pre-filled one-item queue never returns.
- **First writer wins; later writers are dropped silently.** Whether the caller sees audio or `JobCancelled` depends on who got there first, and both outcomes are correct.
- **`cancel()` stays idempotent.** It is already called on paths that may call it twice; a second call must not raise.
- **No behaviour change on the success path.** A job that generates normally must still deliver its array to `result()` exactly as today.
- **No new dependencies.** Standard library only.
- **`JOB_TIMEOUT_S` stays 120.0 and `TTS_THREAD_LIMITER` stays sized to `MAX_STREAMS`.** This plan removes the reason threads pile up rather than papering over it with a shorter timeout.
- Existing suite is **98 passed, 4 deselected**. It must stay green at every commit.

## Background: why the bug is invisible on inspection

`ws_session.run()` already contains `self._pending.cancel()` under a `finally`, with a comment stating the intent. It never executes on the disconnect path:

1. The socket closes. `_read` catches `WebSocketDisconnect` and calls `_abandon()`, which cancels the task group's scope.
2. `_speak` is cancelled inside `await await_job(job)` in `_speak_one`.
3. Python unwinds `_speak_one`'s `finally: self._pending = None` — **inner `finally` runs first**.
4. The cancellation propagates; `run()`'s `finally` tests `if self._pending is not None` and finds `None`.

`streaming_api_omnivoice.text_to_speech_stream` avoids this by assigning `pending = None` on the line *after* `pending.result()` returns, so the assignment is simply never reached when the generator is closed. Task 2 applies the same shape.

Measured before the fix, disconnecting mid-generation with four queued chunks:

```
submitted to scheduler : ['job-0 ...', 'job-1 ...']
generate() actually ran: ['job-0 ...', 'job-1 ...']
job.cancel() called on : []          <- expected job-1
```

## File Structure

| File | Responsibility |
|---|---|
| `batch_scheduler.py` (modify) | Task 1. `JobCancelled` exception, `_deliver` helper, non-blocking `set_result`/`set_exception`, `cancel()` releasing a blocked `result()`. |
| `tests/test_batch_scheduler.py` (modify) | Task 1. Delivery-race and cancel-releases-waiter coverage. |
| `ws_session.py` (modify) | Task 2. `_speak_one` clears `_pending` only on success. |
| `tests/test_ws_speech.py` (modify) | Task 2. Regression test asserting `cancel()` fires on disconnect. |

No files are created. No public signatures change: `cancel()`, `set_result()`, `set_exception()`, and `result()` keep their current names and parameters.

---

### Task 1: Make job delivery non-blocking and first-writer-wins

**Files:**
- Modify: `batch_scheduler.py:19-60` (the `GenerationJob` dataclass)
- Test: `tests/test_batch_scheduler.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces, all in `batch_scheduler`:
  - `class JobCancelled(Exception)` — raised by `result()` when `cancel()` won the race to the slot.
  - `GenerationJob.cancel() -> None` — sets the cancelled flag **and** releases a caller blocked in `result()`.
  - `GenerationJob.set_result(audio: np.ndarray) -> None` — unchanged signature, now non-blocking.
  - `GenerationJob.set_exception(exc: BaseException) -> None` — unchanged signature, now non-blocking.
  - `GenerationJob._deliver(payload: tuple) -> None` — private, first-writer-wins.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_batch_scheduler.py`. Note the existing file already imports
`BatchScheduler, GenerationJob` from `batch_scheduler` at line 6 — extend that
import rather than adding a second one:

```python
from batch_scheduler import BatchScheduler, GenerationJob, JobCancelled
```

Then append these tests:

```python
def test_cancel_releases_a_caller_already_blocked_in_result():
    # The scheduler skips a cancelled job, so nothing will ever fill its slot.
    # Without cancel() writing one, this waiter would block for its whole
    # timeout while holding a worker thread from the TTS limiter.
    job = make_job()
    threading.Timer(0.1, job.cancel).start()

    started = time.monotonic()
    with pytest.raises(JobCancelled):
        job.result(timeout=5)
    assert time.monotonic() - started < 1.0, "cancel did not release the waiter"


def test_cancel_after_a_result_leaves_the_result_intact():
    # A cancel that loses the race must not overwrite real audio with an error.
    job = make_job()
    job.set_result(np.full(4, 7.0, dtype=np.float32))
    job.cancel()
    assert job.result(timeout=1).tolist() == [7.0] * 4


def test_the_worker_never_blocks_when_cancel_filled_the_slot_first():
    # queue.Queue(maxsize=1).put() blocks when full. If set_result still used
    # it, a cancel winning this race would wedge the scheduler's single worker
    # thread forever and every later request would hang.
    job = make_job()
    job.cancel()

    finished = threading.Event()
    threading.Thread(
        target=lambda: (
            job.set_result(np.full(4, 1.0, dtype=np.float32)),
            finished.set(),
        ),
        daemon=True,
    ).start()
    assert finished.wait(timeout=2.0), "set_result blocked on a full slot"


def test_cancel_is_idempotent():
    job = make_job()
    job.cancel()
    job.cancel()
    assert job.cancelled is True


def test_a_cancelled_job_still_reports_itself_cancelled(scheduler, fake_model):
    # The flag the worker checks must survive the new slot write.
    job = make_job(text="job-9")
    job.cancel()
    scheduler.submit(job)
    live = make_job(text="job-1")
    scheduler.submit(live)
    assert live.result(timeout=10).tolist() == [1.0] * 4
    assert job.cancelled is True
    assert all("job-9" not in c["text"] for c in fake_model.calls)
```

These need `threading` and `time` at the top of the file, which it does not
import today. Add them under `from __future__ import annotations`:

```python
import threading
import time
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_batch_scheduler.py -q`
Expected: FAIL at import with `ImportError: cannot import name 'JobCancelled' from 'batch_scheduler'`.

- [ ] **Step 3: Add `JobCancelled` and rewrite delivery**

In `batch_scheduler.py`, add the exception just above the `GenerationJob`
dataclass (after the `COLLECT_WINDOW_S` constant, before `@dataclass`):

```python
class JobCancelled(Exception):
    """Raised by GenerationJob.result() when the job was cancelled first.

    The scheduler drops a cancelled job instead of generating it, so nobody
    would otherwise fill its result slot and the caller would block for the
    whole timeout holding a worker thread.
    """
```

Then replace `cancel`, `set_result`, and `set_exception` (currently
`batch_scheduler.py:47-54`) with:

```python
    def cancel(self) -> None:
        """Mark the job dead and release anyone blocked in result().

        Idempotent: a second call sees a full slot and drops its payload.
        """
        self._cancelled.set()
        self._deliver((None, JobCancelled("generation cancelled")))

    def set_result(self, audio: np.ndarray) -> None:
        self._deliver((audio, None))

    def set_exception(self, exc: BaseException) -> None:
        self._deliver((None, exc))

    def _deliver(self, payload: tuple) -> None:
        """First writer wins; later writers are dropped.

        The slot holds one item, so a blocking put would wedge whichever side
        lost the race — and losing it on the worker thread would take the whole
        scheduler down with it. Every write is therefore non-blocking, and both
        outcomes are correct: the caller gets real audio if generation won, and
        JobCancelled if the cancel did.
        """
        try:
            self._slot.put_nowait(payload)
        except queue.Full:
            pass
```

`result()` is unchanged — it already re-raises whatever exception is in the
slot, so `JobCancelled` propagates without further work.

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_batch_scheduler.py -q`
Expected: PASS, `22 passed` — the file's existing 17 tests plus the 5 new ones.

- [ ] **Step 5: Run the whole suite to prove nothing regressed**

Run: `python -m pytest -q`
Expected: PASS, `103 passed, 4 deselected` — the previous 98 plus Task 1's 5.

The pre-existing `test_cancelled_jobs_are_skipped` is the one to watch: it
cancels before submitting and then asserts a *different* job's result. It never
calls `result()` on the cancelled job, so the new slot write does not affect it.

- [ ] **Step 6: Commit**

```bash
git add batch_scheduler.py tests/test_batch_scheduler.py
git commit -m "fix: cancel() releases a blocked result() without deadlocking the worker"
```

---

### Task 2: Cancel the pending job when the socket disconnects

**Files:**
- Modify: `ws_session.py:316-338` (`WebSocketSpeechSession._speak_one`); the lines to replace are `326-332`
- Test: `tests/test_ws_speech.py`

**Interfaces:**
- Consumes: `GenerationJob.cancel()` from Task 1, which must already release a blocked `result()`. Landing this task without Task 1 leaks worker threads for 120s each.
- Produces: no new names. `_speak_one(chunk: str) -> None` keeps its signature.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ws_speech.py`. The file already imports `json`, `re`,
`threading`, and `time`, and defines `connect()` and `audio_frames()` at the
top — reuse them:

```python
def test_a_disconnect_cancels_the_job_the_session_had_submitted(client, monkeypatch):
    # On a busy server the session's job sits in the scheduler FIFO behind
    # other work. If the client vanishes, that job must be marked cancelled so
    # the worker drops it instead of generating audio for a closed socket.
    import batch_scheduler

    cancelled: list[str] = []
    original = batch_scheduler.GenerationJob.cancel

    def spy(self):
        cancelled.append(self.text)
        return original(self)

    monkeypatch.setattr(batch_scheduler.GenerationJob, "cancel", spy)

    # Generation has to outlast the disconnect, or the job completes before
    # there is anything left to cancel.
    slow = client.fake_model.generate

    def slow_generate(text, **kwargs):
        time.sleep(0.8)
        return slow(text, **kwargs)

    monkeypatch.setattr(client.fake_model, "generate", slow_generate)

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1 is a whole sentence."})
        time.sleep(0.2)  # let the reader submit it, then walk away mid-generation

    assert cancelled, "disconnect left the submitted job uncancelled"
    assert any("job-1" in text for text in cancelled), cancelled


def test_a_normal_finish_cancels_nothing(client, monkeypatch):
    # The counterpart: a session that completes must not cancel the job it
    # already collected, or the fix would be firing on the wrong path.
    import batch_scheduler

    cancelled: list[str] = []
    original = batch_scheduler.GenerationJob.cancel

    def spy(self):
        cancelled.append(self.text)
        return original(self)

    monkeypatch.setattr(batch_scheduler.GenerationJob, "cancel", spy)

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)

    assert final["type"] == "done"
    assert frames
    assert cancelled == [], f"a completed session cancelled a job: {cancelled}"
```

- [ ] **Step 2: Run the tests to verify the first one fails**

Run: `python -m pytest tests/test_ws_speech.py -q -k "cancels"`
Expected: `test_a_disconnect_cancels_the_job_the_session_had_submitted` FAILS
with `AssertionError: disconnect left the submitted job uncancelled`.
`test_a_normal_finish_cancels_nothing` PASSES already — it is a guard against
over-firing, not a driver.

- [ ] **Step 3: Clear `_pending` only on success**

In `ws_session.py`, `_speak_one` currently reads:

```python
        self._pending = job
        self._engine.scheduler.submit(job)
        try:
            audio = await await_job(job)
        finally:
            self._pending = None
```

Replace that block with:

```python
        self._pending = job
        self._engine.scheduler.submit(job)
        # Cleared only after the audio is in hand, exactly as the HTTP path in
        # text_to_speech_stream does it. Clearing in a `finally` would also run
        # while unwinding a cancellation, and because an inner `finally` runs
        # before an outer one, run()'s `finally` would then find _pending
        # already None and never cancel the job — leaving it to generate for a
        # socket that closed.
        audio = await await_job(job)
        self._pending = None
```

Nothing else in the method changes; the lines below it that check
`audio_chunk_is_empty` and encode the audio stay as they are.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ws_speech.py -q -k "cancels"`
Expected: PASS (2 tests).

- [ ] **Step 5: Confirm no session leaks a slot or a thread**

Run: `python -m pytest tests/test_ws_speech.py -q`
Expected: PASS (22 tests). The existing slot-accounting tests —
`test_the_slot_is_returned_when_the_session_ends` and
`test_a_client_disconnecting_mid_session_leaks_no_slot` — are the ones proving
the new `cancel()` call did not change teardown.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS, `105 passed, 4 deselected` — 98 before this plan, plus Task 1's
5 and Task 2's 2.

- [ ] **Step 7: Verify the original symptom is gone**

The bug was found with a spy on `cancel()` and a slow model. Re-run that probe
and confirm the queued chunk is now cancelled rather than generated:

```bash
python - <<'PY'
import sys, time
sys.path.insert(0, "tests")
from fastapi.testclient import TestClient
import batch_scheduler, streaming_api_omnivoice as api
from admission import AdmissionControl
from conftest import FakeOmniVoice

CANCELLED = []
_orig = batch_scheduler.GenerationJob.cancel
def spy(self):
    CANCELLED.append(self.text)
    return _orig(self)
batch_scheduler.GenerationJob.cancel = spy

class SlowFake(FakeOmniVoice):
    def generate(self, text, **kw):
        time.sleep(0.6)
        return super().generate(text, **kw)

fake = SlowFake()
api.service.scheduler = batch_scheduler.BatchScheduler(model_factory=lambda: fake)
api.service.scheduler.start()
assert api.service.scheduler.wait_ready(timeout=5)
api.service.admission = AdmissionControl(max_streams=4)

with TestClient(api.app) as client:
    url = "/v1/audio/speech/ws?voice=af_heart&response_format=pcm"
    with client.websocket_connect(url) as socket:
        for i in range(4):
            socket.send_json({"type": "text", "text": f"job-{i} is a whole sentence."})
        socket.receive()
        print("disconnecting mid-generation")
    time.sleep(2.5)

print("job.cancel() called on:", CANCELLED)
assert CANCELLED, "STILL BROKEN: nothing was cancelled"
print("OK")
api.service.scheduler.stop()
PY
```

Expected: `job.cancel() called on: ['job-1 is a whole sentence.']` then `OK`.
Before the fix this printed an empty list.

- [ ] **Step 8: Commit**

```bash
git add ws_session.py tests/test_ws_speech.py
git commit -m "fix: cancel the pending job when a websocket disconnects"
```

---

## Design notes

**Why not shorten `JOB_TIMEOUT_S` instead.** A blocked waiter on a skipped job
is the symptom; the missing slot write is the cause. Dropping the timeout to,
say, 5s would shrink the thread leak but would also abort legitimately slow
generations under load, and the leak would still exist. Task 1 removes the wait
entirely.

**Why `JobCancelled` rather than reusing `queue.Empty` or returning `None`.**
`result()` already re-raises whatever is in the slot, so an exception costs no
new plumbing. `None` would force every caller to branch, and `queue.Empty`
already means "timed out", which is a different and genuinely worse condition
that should stay distinguishable in logs.

**Where `JobCancelled` actually surfaces.** Only on an abandoned worker thread.
`cancel()` is called from `run()`'s `finally`, which runs after the task group
has already unwound `_speak`, so no live session ever catches it. Verified that
an abandoned `anyio.to_thread.run_sync(..., abandon_on_cancel=True)` thread
raising produces no warnings and does not disturb the event loop. `_speak`'s
`except Exception` therefore does not need a `JobCancelled` branch — adding one
would be dead code.

**Why the HTTP path needs no change.** `text_to_speech_stream` already assigns
`pending = None` after `result()` returns rather than in a `finally`, so it has
always cancelled correctly on `GeneratorExit`. Task 1 does change what its
`pending.cancel()` does — it now also writes to a slot nobody is reading, which
is harmless.

## Out of scope

Recorded so they are choices, not oversights:

- **Barge-in / `{"type":"cancel"}`.** Decided against: the client will close and
  reopen the connection instead. That makes disconnect a hot path, which is what
  raises the value of this fix, but it needs no protocol change.
- **Stopping a job already inside `model.generate()`.** Impossible without
  step-wise execution in the model. One chunk of GPU work per disconnect stays
  wasted, and no design in reach changes that.
- **Cancelling from the reader while the speaker is mid-await.** Only needed for
  barge-in, which is out of scope. `cancel()` is called from `run()`'s teardown
  and nowhere else.
- **Metrics on cancelled jobs.** No counter for how often the scheduler skips a
  cancelled job. Worth adding if you later want to measure disconnect waste.
