# In-Process Batch Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `/api/stream-mp3` serve several simultaneous streams by batching their text chunks into single `OmniVoice.generate()` calls, instead of serializing every request behind one global lock.

**Architecture:** A new `batch_scheduler.py` owns the `OmniVoice` model and exactly one worker thread. Request threads submit one text chunk at a time as a `GenerationJob` and block on its result; the worker collects compatible jobs over a short window and issues one batched `generate()` call with per-item lists. Because the worker is the sole toucher of the model *and* the voice-prompt cache, `OmniVoiceStreamingService._lock` is deleted rather than narrowed.

**Tech Stack:** Python 3.10, FastAPI 0.135.3, `omnivoice`, PyTorch 2.8 (CUDA 12.8), `lameenc`, `pytest` (new dev dependency).

## Why in-process rather than vLLM

vLLM-Omni was investigated first and rejected: it serves OmniVoice strictly one request at a time, with no way to enable request batching. See the note in [README.md](README.md#note-vllm-omni-cannot-serve-omnivoice-concurrently). The standalone `omnivoice` package batches natively, so this scheduler can do the one thing vLLM cannot.

## Global Constraints

- Python 3.10 — use `X | None` unions (already the file's style).
- `from __future__ import annotations` at the top of every new module, matching `streaming_api_omnivoice.py`.
- The public HTTP contract of `/api/stream-mp3` (POST and GET) does not change: same request fields, same `audio/mpeg` streaming response, same headers.
- Only voice-**cloning** mode is supported. Every request resolves a reference audio via `get_voice_clone_config`, which either returns `(path, ref_text)` or raises 404/400, so `voice_clone_prompt` is always present. The `elif request.instruct` branch at `streaming_api_omnivoice.py:342` is already unreachable and is removed by this plan.
- **Batch grouping key is `(num_step, denoise, postprocess_output)`.** Those three are per-**batch** fields of `OmniVoiceGenerationConfig`; `text`, `language`, `speed`, and `voice_clone_prompt` are per-**item** and may differ freely inside one batch. Different voices batch together fine.
- Never call `OmniVoice.generate`, `OmniVoice.from_pretrained`, or `create_voice_clone_prompt` from any thread other than the scheduler worker.
- Defaults, overridable by environment variable:
  - `OMNIVOICE_MAX_BATCH` = `4`
  - `OMNIVOICE_COLLECT_WINDOW_MS` = `10`
  - `OMNIVOICE_MAX_STREAMS` = `16`
- Tests must not require a GPU or download model weights, except those marked `@pytest.mark.gpu`.

## Facts verified in the `omnivoice` source

These drove the design; re-check them if the package is upgraded (currently 0.1.5 installed, 0.1.3 pinned in `requirements.txt`).

- `generate()` accepts `text: Union[str, list[str]]` with per-item `language`, `speed`, `voice_clone_prompt`, `instruct` (`models/omnivoice.py:476`).
- Per-item reference data is threaded through by index: `ref_text_list`, `ref_audio_tokens_list`, `ref_rms_list` are built from each item's own prompt (`models/omnivoice.py:961-966`), then consumed per-index when building inputs (`:1173-1183`) and again on decode (`ref_rms[i]`). **So different voices in one batch are safe.**
- **But mode is decided by item 0 only:** `if voice_clone_prompt_list[0] is not None:` (`:961`). Mixing a cloning item with a non-cloning item in one batch either crashes on `vc.ref_text` or silently drops the cloning. Irrelevant here because all traffic is cloning — but it is why mode must never become a per-item concern.
- Classifier-free guidance allocates every tensor at `2 * B` — conditional rows `0..B-1`, unconditional `B..2B-1` (`:1190-1213`). A batch of 4 is 8 rows through the transformer.
- The attention mask is dense and quadratic: `(2B, 1, max_c_len, max_c_len)` (`:1198-1200`). Bounded only by `chunk_chars`; do not raise that ceiling without re-measuring VRAM.
- Inputs pad to `max_c_len` and `max(target_lens)` (`:1186-1223`), so mixing very short and very long chunks in one batch wastes compute.
- `get_indices` splits a batch at a 30-second threshold (`:130-134`). Chunked text stays well under it, so batches will not fragment.

---

## File Structure

| File | Responsibility |
|---|---|
| `batch_scheduler.py` *(create)* | `GenerationJob`, `BatchScheduler`. Owns the model + worker thread. No HTTP, no MP3, no FastAPI imports. |
| `admission.py` *(create)* | `AdmissionControl`, `StreamSlot`. Bounded concurrent-stream counter with idempotent release. |
| `streaming_api_omnivoice.py` *(modify)* | Loses `_lock`, `_get_model`, `_get_voice_clone_prompt`, and the direct `model.generate` call. Gains scheduler submission and 503 admission. |
| `tests/conftest.py` *(create)* | `FakeOmniVoice` fixture and the `gpu` marker. |
| `tests/test_batch_scheduler.py` *(create)* | Batching policy, result routing, error isolation, cancellation. |
| `tests/test_admission.py` *(create)* | 503 behaviour and slot-leak prevention. |
| `tests/test_streaming_integration.py` *(create)* | `@pytest.mark.gpu` concurrent end-to-end streams. |
| `bench/bench_batch.py` *(create)* | Latency + peak VRAM at batch 1/2/4/8. Not a test. |

See `TESTING.md` for how to run each layer.

---

### Task 0: Capture the "before" baseline — RUN THIS FIRST

**Files:**
- Create: `bench/bench_concurrent.py` *(already written)*
- Create: `bench/results/before.json` *(generated)*

**Interfaces:**
- Consumes: the current, unmodified API.
- Produces: `bench/results/before.json`, the baseline every later comparison is measured against.

**This task must complete before Task 5 touches `streaming_api_omnivoice.py`.** Once the lock is gone the "before" state is unrecoverable without a `git stash`, so capture it now. Tasks 1-4 only add new files and are safe to do first if you prefer, but there is no reason to wait.

The harness fires 8 concurrent `/api/stream-mp3` requests and records per-request time-to-first-byte and total time, plus a single-request serial reference. The headline metric is `speedup_vs_serial` — how much faster 8 concurrent requests are than the same 8 run back to back. **A fully serialized server scores ~1.0x regardless of how fast it is**, which is exactly what the current global lock should produce. That makes it a clean before/after discriminator: raw latency depends on the GPU's mood, but the ratio does not.

- [ ] **Step 1: Confirm the GPU is free**

Run: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
Expected: no processes holding significant memory. A Jupyter kernel from `~/huggingface/qwen3-asr-lora` has previously held ~12 GB of the 16.3 GB card.

- [ ] **Step 2: Start the API on the unmodified code**

In a separate terminal:

```bash
venv/bin/uvicorn streaming_api_omnivoice:app --host 0.0.0.0 --port 9000
```

Wait for startup warmup to finish. Confirm: `curl -s localhost:9000/health`

- [ ] **Step 3: Capture the baseline**

```bash
venv/bin/python bench/bench_concurrent.py --label before
```

Expected: `succeeded 8/8`, and `speedup_vs_serial` close to **1.0x**. Written to `bench/results/before.json`.

If the speedup is already well above 1.5x, stop — the premise is wrong and something is not serializing the way the code reads. Investigate before building anything.

- [ ] **Step 4: Sanity-check the saved record**

```bash
venv/bin/python -c "
import json; d=json.load(open('bench/results/before.json'))
print('speedup', d['speedup_vs_serial'], '| ok', d['succeeded'], '| wall', d['concurrent_wall_s'])
assert d['succeeded'] == 8, d['failed']
assert d['ttfb_median'] is not None
print('baseline usable')"
```

Expected: prints `baseline usable`. A run with failures is not a baseline — fix the failure and re-run before continuing.

- [ ] **Step 5: Commit the baseline**

```bash
git add bench/bench_concurrent.py bench/results/before.json
git commit -m "bench: capture pre-scheduler concurrency baseline"
```

Committing the JSON is deliberate — it is evidence, and it records the commit the measurement was taken at.

---

### Task 1: Test harness and `GenerationJob`

**Files:**
- Create: `batch_scheduler.py`
- Create: `tests/conftest.py`
- Create: `tests/test_batch_scheduler.py`
- Modify: `requirements.txt`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: `GenerationJob(text: str, ref_audio: str, ref_text: str, language: str | None, speed: float, num_step: int, denoise: bool, postprocess_output: bool)` with `.batch_key -> tuple[int, bool, bool]`, `.set_result(np.ndarray) -> None`, `.set_exception(BaseException) -> None`, `.result(timeout: float | None = None) -> np.ndarray`, `.cancel() -> None`, `.cancelled -> bool`. Also `FakeOmniVoice` test double.

- [ ] **Step 1: Add dev dependencies and pytest config**

Append to `requirements.txt`:

```
## Dev / test only
# pip install pytest==8.3.4 pytest-timeout==2.3.1
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
timeout = 60
markers =
    gpu: requires a CUDA GPU and downloaded OmniVoice weights
addopts = -m "not gpu"
```

Install: `venv/bin/pip install pytest==8.3.4 pytest-timeout==2.3.1`

- [ ] **Step 2: Write the `FakeOmniVoice` double**

Create `tests/conftest.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


class FakeOmniVoice:
    """Stand-in for OmniVoice that records how it was called.

    generate() returns, for each input text "job-<n>", an array filled with
    float(n). That lets tests assert a result reached the job that asked for
    it, which is the failure mode that would swap audio between users.
    """

    sampling_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_texts: set[str] = set()
        self.prompt_calls: list[tuple[str, str]] = []

    def create_voice_clone_prompt(self, ref_audio, ref_text, preprocess_prompt=True):
        self.prompt_calls.append((ref_audio, ref_text))
        return f"prompt:{ref_audio}"

    def generate(self, text, **kwargs):
        self.calls.append({"text": list(text), **kwargs})
        failing = [t for t in text if t in self.fail_texts]
        if failing:
            raise RuntimeError(f"generate failed for {failing}")
        return [np.full(4, float(t.split("-")[1]), dtype=np.float32) for t in text]


@pytest.fixture
def fake_model() -> FakeOmniVoice:
    return FakeOmniVoice()
```

- [ ] **Step 3: Write the failing test for `GenerationJob`**

Create `tests/test_batch_scheduler.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from batch_scheduler import GenerationJob


def make_job(text="job-0", num_step=16, denoise=False, postprocess_output=False):
    return GenerationJob(
        text=text,
        ref_audio="voices/af_heart.mp3",
        ref_text="reference transcript",
        language=None,
        speed=0.9,
        num_step=num_step,
        denoise=denoise,
        postprocess_output=postprocess_output,
    )


def test_batch_key_covers_per_batch_config_only():
    a = make_job(text="job-0")
    b = make_job(text="job-1")
    assert a.batch_key == b.batch_key == (16, False, False)


def test_batch_key_differs_when_num_step_differs():
    assert make_job(num_step=16).batch_key != make_job(num_step=32).batch_key


def test_result_returns_what_was_set():
    job = make_job()
    job.set_result(np.full(4, 7.0, dtype=np.float32))
    assert job.result(timeout=1).tolist() == [7.0, 7.0, 7.0, 7.0]


def test_result_reraises_the_exception_that_was_set():
    job = make_job()
    job.set_exception(RuntimeError("generate blew up"))
    with pytest.raises(RuntimeError, match="generate blew up"):
        job.result(timeout=1)


def test_cancel_marks_job_cancelled():
    job = make_job()
    assert job.cancelled is False
    job.cancel()
    assert job.cancelled is True
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_batch_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'batch_scheduler'`

- [ ] **Step 5: Implement `GenerationJob`**

Create `batch_scheduler.py`:

```python
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

MAX_BATCH = int(os.environ.get("OMNIVOICE_MAX_BATCH", "4"))
COLLECT_WINDOW_S = float(os.environ.get("OMNIVOICE_COLLECT_WINDOW_MS", "10")) / 1000.0


@dataclass
class GenerationJob:
    """One text chunk awaiting generation.

    Exactly one of set_result/set_exception is called by the worker thread;
    the submitting thread blocks in result() until then.
    """

    text: str
    ref_audio: str
    ref_text: str
    language: str | None
    speed: float
    num_step: int
    denoise: bool
    postprocess_output: bool

    _slot: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))
    _cancelled: threading.Event = field(default_factory=threading.Event)

    @property
    def batch_key(self) -> tuple[int, bool, bool]:
        return (self.num_step, self.denoise, self.postprocess_output)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def set_result(self, audio: np.ndarray) -> None:
        self._slot.put((audio, None))

    def set_exception(self, exc: BaseException) -> None:
        self._slot.put((None, exc))

    def result(self, timeout: float | None = None) -> np.ndarray:
        audio, exc = self._slot.get(timeout=timeout)
        if exc is not None:
            raise exc
        return audio
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_batch_scheduler.py -v`
Expected: PASS — 5 passed

- [ ] **Step 7: Commit**

```bash
git add batch_scheduler.py tests/conftest.py tests/test_batch_scheduler.py pytest.ini requirements.txt
git commit -m "feat: add GenerationJob and test harness for batch scheduler"
```

---

### Task 2: `BatchScheduler` batching and result routing

**Files:**
- Modify: `batch_scheduler.py`
- Modify: `tests/test_batch_scheduler.py`

**Interfaces:**
- Consumes: `GenerationJob` from Task 1; `FakeOmniVoice` from `tests/conftest.py`.
- Produces: `BatchScheduler(model_factory: Callable[[], object], max_batch: int = MAX_BATCH, collect_window_s: float = COLLECT_WINDOW_S)` with `.start() -> None`, `.stop() -> None`, `.submit(job: GenerationJob) -> None`, `.wait_ready(timeout: float) -> bool`, `.sampling_rate -> int`.

- [ ] **Step 1: Write the failing batching tests**

Append to `tests/test_batch_scheduler.py`:

```python
from batch_scheduler import BatchScheduler


@pytest.fixture
def scheduler(fake_model):
    sched = BatchScheduler(model_factory=lambda: fake_model, max_batch=4)
    sched.start()
    assert sched.wait_ready(timeout=5)
    yield sched
    sched.stop()


def submit_all(scheduler, jobs):
    for job in jobs:
        scheduler.submit(job)
    return [job.result(timeout=10) for job in jobs]


def test_single_job_round_trips(scheduler, fake_model):
    job = make_job(text="job-3")
    scheduler.submit(job)
    assert job.result(timeout=10).tolist() == [3.0] * 4
    assert len(fake_model.calls) == 1


def test_results_are_routed_to_the_job_that_asked_for_them(scheduler):
    jobs = [make_job(text=f"job-{i}") for i in range(4)]
    results = submit_all(scheduler, jobs)
    for i, result in enumerate(results):
        assert result.tolist() == [float(i)] * 4


def test_jobs_sharing_a_batch_key_ride_one_generate_call(scheduler, fake_model):
    jobs = [make_job(text=f"job-{i}") for i in range(4)]
    submit_all(scheduler, jobs)
    batched = [c for c in fake_model.calls if len(c["text"]) > 1]
    assert batched, f"expected a batched call, got {fake_model.calls}"


def test_batch_never_exceeds_max_batch(scheduler, fake_model):
    jobs = [make_job(text=f"job-{i}") for i in range(10)]
    submit_all(scheduler, jobs)
    assert max(len(c["text"]) for c in fake_model.calls) <= 4


def test_different_batch_keys_are_never_mixed(scheduler, fake_model):
    jobs = [make_job(text=f"job-{i}", num_step=16 if i % 2 == 0 else 32) for i in range(6)]
    submit_all(scheduler, jobs)
    for call in fake_model.calls:
        steps = {16 if int(t.split("-")[1]) % 2 == 0 else 32 for t in call["text"]}
        assert len(steps) == 1, f"mixed num_step in one call: {call}"
        assert call["num_step"] == steps.pop()


def test_per_item_arguments_are_passed_as_aligned_lists(scheduler, fake_model):
    jobs = [make_job(text=f"job-{i}") for i in range(3)]
    submit_all(scheduler, jobs)
    call = max(fake_model.calls, key=lambda c: len(c["text"]))
    n = len(call["text"])
    assert len(call["speed"]) == n
    assert len(call["language"]) == n
    assert len(call["voice_clone_prompt"]) == n


def test_voice_clone_prompts_are_cached_per_reference(scheduler, fake_model):
    submit_all(scheduler, [make_job(text=f"job-{i}") for i in range(4)])
    assert len(fake_model.prompt_calls) == 1


def test_wait_ready_is_false_when_the_model_fails_to_load():
    def explode():
        raise RuntimeError("no weights on disk")

    sched = BatchScheduler(model_factory=explode)
    sched.start()
    # Must be False, not True-with-a-dead-worker: warmup relies on this to
    # avoid submitting a job nobody will ever answer.
    assert sched.wait_ready(timeout=5) is False
    sched.stop()


def test_cancelled_jobs_are_skipped(scheduler, fake_model):
    cancelled = make_job(text="job-9")
    cancelled.cancel()
    scheduler.submit(cancelled)
    live = make_job(text="job-1")
    scheduler.submit(live)
    assert live.result(timeout=10).tolist() == [1.0] * 4
    assert all("job-9" not in c["text"] for c in fake_model.calls)
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_batch_scheduler.py -v`
Expected: FAIL — `ImportError: cannot import name 'BatchScheduler'`

- [ ] **Step 3: Implement `BatchScheduler`**

Append to `batch_scheduler.py`:

```python
class BatchScheduler:
    """Owns the model and the single thread allowed to touch it.

    Callers submit one job at a time and block on job.result(). The worker
    groups jobs sharing a batch_key into one generate() call.
    """

    def __init__(
        self,
        model_factory: Callable[[], object],
        max_batch: int = MAX_BATCH,
        collect_window_s: float = COLLECT_WINDOW_S,
    ) -> None:
        self._model_factory = model_factory
        self._max_batch = max_batch
        self._collect_window_s = collect_window_s
        self._queue: queue.Queue[GenerationJob] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._model = None
        self._prompt_cache: dict[tuple[str, str], object] = {}
        self.sampling_rate = 0

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="omnivoice-batch-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def wait_ready(self, timeout: float) -> bool:
        """True only if the model actually loaded.

        The ready event fires on failure too, so callers must not treat it
        as success — otherwise warmup would submit a job to a dead worker
        and block forever on a slot nobody will ever fill.
        """
        return self._ready.wait(timeout) and self._model is not None

    def submit(self, job: GenerationJob) -> None:
        self._queue.put(job)

    # -- worker --------------------------------------------------------

    def _run(self) -> None:
        try:
            self._model = self._model_factory()
            self.sampling_rate = self._model.sampling_rate
        except Exception:
            logger.exception("failed to load OmniVoice model")
            self._ready.set()
            return
        self._ready.set()

        while not self._stopping.is_set():
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
            except Exception:
                # The worker must outlive any single failure; if it dies,
                # every future request hangs forever waiting on its slot.
                logger.exception("batch worker loop error")

    def _collect_batch(self) -> list[GenerationJob]:
        first: GenerationJob | None = None
        while first is None:
            if self._stopping.is_set():
                return []
            try:
                candidate = self._queue.get(timeout=0.1)
            except queue.Empty:
                return []
            if not candidate.cancelled:
                first = candidate

        batch = [first]
        key = first.batch_key
        deferred: list[GenerationJob] = []
        deadline = time.monotonic() + self._collect_window_s

        while len(batch) < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                job = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if job.cancelled:
                continue
            if job.batch_key == key:
                batch.append(job)
            else:
                deferred.append(job)

        for job in deferred:
            self._queue.put(job)
        return batch

    def _voice_clone_prompt(self, ref_audio: str, ref_text: str):
        key = (ref_audio, ref_text)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                preprocess_prompt=True,
            )
        return self._prompt_cache[key]

    def _generate(self, batch: list[GenerationJob]) -> list[np.ndarray]:
        head = batch[0]
        return self._model.generate(
            text=[j.text for j in batch],
            language=[j.language for j in batch],
            speed=[j.speed for j in batch],
            voice_clone_prompt=[
                self._voice_clone_prompt(j.ref_audio, j.ref_text) for j in batch
            ],
            num_step=head.num_step,
            denoise=head.denoise,
            postprocess_output=head.postprocess_output,
        )

    def _process_batch(self, batch: list[GenerationJob]) -> None:
        audios = self._generate(batch)
        for job, audio in zip(batch, audios):
            job.set_result(audio)
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest tests/test_batch_scheduler.py -v`
Expected: PASS — all tests pass

- [ ] **Step 5: Commit**

```bash
git add batch_scheduler.py tests/test_batch_scheduler.py
git commit -m "feat: batch scheduler groups compatible jobs into one generate call"
```

---

### Task 3: Fault isolation — one bad chunk must not kill its batchmates

**Files:**
- Modify: `batch_scheduler.py` (replace `_process_batch`)
- Modify: `tests/test_batch_scheduler.py`

**Interfaces:**
- Consumes: `BatchScheduler._generate` from Task 2.
- Produces: no new public API. `_process_batch` gains individual-retry behaviour.

Rationale: a batched `generate()` raises for the whole batch, so one malformed chunk would fail three innocent streams. On batch failure the worker retries each item alone, so only the genuinely bad item errors.

- [ ] **Step 1: Write the failing isolation tests**

Append to `tests/test_batch_scheduler.py`:

```python
def test_one_poison_chunk_does_not_fail_its_batchmates(scheduler, fake_model):
    fake_model.fail_texts = {"job-2"}
    jobs = [make_job(text=f"job-{i}") for i in range(4)]
    for job in jobs:
        scheduler.submit(job)

    for i, job in enumerate(jobs):
        if i == 2:
            with pytest.raises(RuntimeError):
                job.result(timeout=10)
        else:
            assert job.result(timeout=10).tolist() == [float(i)] * 4


def test_single_job_failure_propagates_without_retry_storm(scheduler, fake_model):
    fake_model.fail_texts = {"job-0"}
    job = make_job(text="job-0")
    scheduler.submit(job)
    with pytest.raises(RuntimeError):
        job.result(timeout=10)
    assert len(fake_model.calls) == 1


def test_worker_survives_a_failure_and_serves_the_next_job(scheduler, fake_model):
    fake_model.fail_texts = {"job-0"}
    bad = make_job(text="job-0")
    scheduler.submit(bad)
    with pytest.raises(RuntimeError):
        bad.result(timeout=10)

    fake_model.fail_texts = set()
    good = make_job(text="job-5")
    scheduler.submit(good)
    assert good.result(timeout=10).tolist() == [5.0] * 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_batch_scheduler.py -v`
Expected: FAIL — `test_one_poison_chunk_does_not_fail_its_batchmates` times out, because `_process_batch` lets the exception escape and nobody fills the other jobs' slots.

- [ ] **Step 3: Replace `_process_batch` with the isolating version**

In `batch_scheduler.py`, replace the whole `_process_batch` method with:

```python
    def _process_batch(self, batch: list[GenerationJob]) -> None:
        try:
            audios = self._generate(batch)
        except Exception as exc:
            if len(batch) == 1:
                batch[0].set_exception(exc)
                return
            logger.warning(
                "batch of %d failed (%s); retrying each item alone", len(batch), exc
            )
            for job in batch:
                try:
                    job.set_result(self._generate([job])[0])
                except Exception as item_exc:  # noqa: BLE001 - reported to caller
                    job.set_exception(item_exc)
            return

        for job, audio in zip(batch, audios):
            job.set_result(audio)
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest tests/test_batch_scheduler.py -v`
Expected: PASS — all tests pass

- [ ] **Step 5: Commit**

```bash
git add batch_scheduler.py tests/test_batch_scheduler.py
git commit -m "feat: isolate batch failures by retrying items individually"
```

---

### Task 4: Admission control

**Files:**
- Create: `admission.py`
- Create: `tests/test_admission.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AdmissionControl(max_streams: int = MAX_STREAMS)` with `.try_acquire() -> StreamSlot | None` and `.active -> int`; `StreamSlot` with `.release() -> None` (idempotent).

Rationale: `max_streams` is **not** the batch size. Streams beyond `max_batch` are admitted and simply wait their turn in the scheduler queue; only past `max_streams` does the server refuse with 503. Release must be idempotent because the streaming generator's `finally` and the endpoint's error path can both fire.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_admission.py`:

```python
from __future__ import annotations

import threading

from admission import AdmissionControl


def test_acquires_up_to_the_cap():
    control = AdmissionControl(max_streams=2)
    assert control.try_acquire() is not None
    assert control.try_acquire() is not None
    assert control.try_acquire() is None


def test_releasing_frees_a_slot():
    control = AdmissionControl(max_streams=1)
    slot = control.try_acquire()
    assert control.try_acquire() is None
    slot.release()
    assert control.try_acquire() is not None


def test_release_is_idempotent():
    control = AdmissionControl(max_streams=1)
    slot = control.try_acquire()
    slot.release()
    slot.release()
    slot.release()
    assert control.active == 0
    assert control.try_acquire() is not None


def test_active_count_tracks_outstanding_slots():
    control = AdmissionControl(max_streams=4)
    slots = [control.try_acquire() for _ in range(3)]
    assert control.active == 3
    slots[0].release()
    assert control.active == 2


def test_concurrent_acquire_never_exceeds_the_cap():
    control = AdmissionControl(max_streams=5)
    acquired = []
    lock = threading.Lock()

    def worker():
        slot = control.try_acquire()
        if slot is not None:
            with lock:
                acquired.append(slot)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(acquired) == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_admission.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'admission'`

- [ ] **Step 3: Implement `admission.py`**

Create `admission.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest tests/test_admission.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add admission.py tests/test_admission.py
git commit -m "feat: add bounded admission control for concurrent streams"
```

---

### Task 5: Wire the scheduler into the streaming service

**Files:**
- Modify: `streaming_api_omnivoice.py:51-111` (remove `_lock`, `_get_model`, `_get_voice_clone_prompt`)
- Modify: `streaming_api_omnivoice.py:292-357` (`text_to_speech_stream`)
- Modify: `streaming_api_omnivoice.py:363-381` (startup, health)

**Interfaces:**
- Consumes: `BatchScheduler`, `GenerationJob` (Tasks 1-3); `StreamSlot` (Task 4).
- Produces: `OmniVoiceStreamingService.text_to_speech_stream(request: StreamRequest, slot: StreamSlot | None = None) -> Generator[bytes, None, None]`.

- [ ] **Step 1: Replace the model-owning members**

Add near the other imports in `streaming_api_omnivoice.py`:

```python
from admission import AdmissionControl
from batch_scheduler import BatchScheduler, GenerationJob
```

Replace `OmniVoiceStreamingService.__init__`, `_get_model`, `warmup`, and `_get_voice_clone_prompt` (lines 52-111) with:

```python
    def __init__(self) -> None:
        self.scheduler = BatchScheduler(model_factory=self._load_model)
        self.admission = AdmissionControl()

    @staticmethod
    def _load_model() -> OmniVoice:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        return OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)

    def warmup(self) -> None:
        if self.scheduler.sampling_rate > 0:
            return

        self.scheduler.start()
        if not self.scheduler.wait_ready(timeout=600):
            raise RuntimeError("OmniVoice model failed to load within 600s")

        try:
            ref_audio_path = self._resolve_ref_audio_path(WARMUP_REF_AUDIO)
        except FileNotFoundError:
            logger.warning(
                "Warmup reference audio not found at %s; skipping warmup generation",
                WARMUP_REF_AUDIO,
            )
            return

        if ref_audio_path is None:
            return

        job = GenerationJob(
            text=WARMUP_TEXT,
            ref_audio=str(ref_audio_path),
            ref_text=WARMUP_REF_TEXT,
            language=None,
            speed=1.0,
            num_step=16,
            denoise=False,
            postprocess_output=False,
        )
        self.scheduler.submit(job)
        job.result(timeout=600)
```

Change `health()` (line 378) — it reads `service._model`, which no longer exists — to:

```python
        "model_loaded": service.scheduler.sampling_rate > 0,
```

- [ ] **Step 2: Rewrite `text_to_speech_stream` to submit jobs**

Replace lines 292-357 with:

```python
    def text_to_speech_stream(
        self,
        request: StreamRequest,
        slot=None,
    ) -> Generator[bytes, None, None]:
        """Stream MP3 audio bytes generated from OmniVoice text chunks."""
        if lameenc is None:
            raise RuntimeError("lameenc is required for MP3 streaming")

        pending: GenerationJob | None = None
        try:
            text = request.text.strip()
            if not text:
                return

            ref_audio_path, ref_text = self.get_voice_clone_config(request.voice_id)

            speed = request.speed if request.speed is not None else 0.8
            if len(text.split()) <= 4:
                speed = 1.0

            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(self.scheduler.sampling_rate)
            encoder.set_channels(1)
            encoder.set_quality(2)

            for text_chunk in self._split_text_for_streaming(
                text,
                max_chars=request.chunk_chars,
            ):
                pending = GenerationJob(
                    text=text_chunk,
                    ref_audio=str(ref_audio_path),
                    ref_text=ref_text,
                    language=request.language,
                    speed=speed,
                    num_step=request.num_step,
                    denoise=request.denoise,
                    postprocess_output=request.postprocess_output,
                )
                self.scheduler.submit(pending)
                audio_chunk = pending.result()
                pending = None

                if self._audio_chunk_is_empty(audio_chunk):
                    continue

                mp3_chunk = encoder.encode(
                    self._audio_tensor_to_int16_bytes(audio_chunk)
                )
                if mp3_chunk:
                    yield bytes(mp3_chunk)

            final_chunk = encoder.flush()
            if final_chunk:
                yield bytes(final_chunk)
        finally:
            # Reached on normal completion and on GeneratorExit when the
            # client disconnects mid-stream.
            if pending is not None:
                pending.cancel()
            if slot is not None:
                slot.release()
```

- [ ] **Step 3: Remove the dead `instruct` branch and unused import**

The `elif request.instruct:` branch is gone with the rewrite above. Remove `import threading` from the top of the file — nothing in `streaming_api_omnivoice.py` uses it any more.

- [ ] **Step 4: Verify the lock and direct model access are gone**

Run:

```bash
venv/bin/python - <<'PY'
import ast
src = open("streaming_api_omnivoice.py").read()
ast.parse(src)
for banned in ("_lock", "model.generate", "import threading"):
    assert banned not in src, f"still present: {banned}"
print("ok")
PY
```

Expected: prints `ok`

- [ ] **Step 5: Commit**

```bash
git add streaming_api_omnivoice.py
git commit -m "refactor: route streaming chunks through the batch scheduler"
```

---

### Task 6: 503 on overload at the endpoint

**Files:**
- Modify: `streaming_api_omnivoice.py:403-418` (`stream_mp3_audio`)
- Modify: `tests/test_admission.py`

**Interfaces:**
- Consumes: `AdmissionControl.try_acquire` (Task 4), `text_to_speech_stream(request, slot)` (Task 5).
- Produces: `/api/stream-mp3` returns `503` with a `Retry-After: 5` header when `max_streams` is exhausted.

- [ ] **Step 1: Write the failing endpoint tests**

Append to `tests/test_admission.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """API client with the model replaced by a stub, so no GPU is needed."""
    import batch_scheduler
    import streaming_api_omnivoice as api
    from conftest import FakeOmniVoice

    fake = FakeOmniVoice()
    api.service.scheduler = batch_scheduler.BatchScheduler(model_factory=lambda: fake)
    api.service.scheduler.start()
    assert api.service.scheduler.wait_ready(timeout=5)
    api.service.admission = AdmissionControl(max_streams=1)

    with TestClient(api.app) as test_client:
        yield test_client
    api.service.scheduler.stop()


def test_returns_503_when_no_slot_is_available(client):
    import streaming_api_omnivoice as api

    held = api.service.admission.try_acquire()
    assert held is not None

    response = client.post(
        "/api/stream-mp3", json={"text": "job-1", "voice_id": "af_heart"}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"

    held.release()


def test_slot_is_returned_after_a_stream_completes(client):
    import streaming_api_omnivoice as api

    response = client.post(
        "/api/stream-mp3", json={"text": "job-1", "voice_id": "af_heart"}
    )
    assert response.status_code == 200
    _ = response.content
    assert api.service.admission.active == 0


def test_unknown_voice_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    response = client.post(
        "/api/stream-mp3", json={"text": "job-1", "voice_id": "nope"}
    )
    assert response.status_code in (404, 500)
    assert api.service.admission.active == 0
```

The `warmup` guard added in Task 5 Step 1 (`if self.scheduler.sampling_rate > 0: return`) is what stops the app's startup event from replacing the stub scheduler this fixture installs.

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_admission.py -v`
Expected: FAIL — `test_returns_503_when_no_slot_is_available` gets `200`, because nothing checks admission yet.

- [ ] **Step 3: Add admission to the endpoint**

Replace `stream_mp3_audio` (lines 403-418) with:

```python
@app.post("/api/stream-mp3")
def stream_mp3_audio(request: StreamRequest) -> StreamingResponse:
    if lameenc is None:
        raise HTTPException(
            status_code=500,
            detail="Install lameenc in the venv to use MP3 streaming",
        )

    slot = service.admission.try_acquire()
    if slot is None:
        raise HTTPException(
            status_code=503,
            detail="Too many concurrent streams; retry shortly",
            headers={"Retry-After": "5"},
        )

    try:
        # Resolve the voice before streaming starts so a bad voice_id is a
        # clean 404 rather than a truncated audio stream.
        service.get_voice_clone_config(request.voice_id)
    except HTTPException:
        slot.release()
        raise

    return StreamingResponse(
        service.text_to_speech_stream(request, slot=slot),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Content-Disposition": 'inline; filename="omnivoice-stream.mp3"',
        },
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS — all non-GPU tests pass

- [ ] **Step 5: Commit**

```bash
git add streaming_api_omnivoice.py tests/test_admission.py
git commit -m "feat: reject overload with 503 instead of unbounded queueing"
```

---

### Task 7: Benchmark and GPU integration test

**Files:**
- Create: `bench/bench_batch.py`
- Create: `tests/test_streaming_integration.py`
- Modify: `TESTING.md`

**Interfaces:**
- Consumes: `BatchScheduler`, `GenerationJob`, the running FastAPI app.
- Produces: a printed table of latency and peak VRAM per batch size; a `@pytest.mark.gpu` concurrency test.

This is where the 2.5-3.5x throughput estimate and the ~4.5-6 GB VRAM estimate get confirmed or corrected.

- [ ] **Step 1: Write the benchmark**

Create `bench/bench_batch.py`:

```python
"""Measure generate() latency and peak VRAM across batch sizes.

Run:  venv/bin/python bench/bench_batch.py
"""
from __future__ import annotations

import time

import torch

from batch_scheduler import BatchScheduler, GenerationJob

REF_AUDIO = "voices/af_heart.mp3"
REF_TEXT = (
    "Human just went farther from earth than ever before. "
    "This was the mission to go back to the moon."
)
CHUNK = (
    "The quick brown fox jumps over the lazy dog while the sun sets "
    "slowly behind the distant mountains and the river runs on."
)


def load_model():
    from omnivoice import OmniVoice

    return OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )


def main() -> None:
    scheduler = BatchScheduler(
        model_factory=load_model, max_batch=64, collect_window_s=0.05
    )
    scheduler.start()
    assert scheduler.wait_ready(timeout=900)

    print(f"{'batch':>6} {'wall_s':>8} {'s/item':>8} {'speedup':>8} {'peak_GB':>9}")
    baseline = None

    for batch_size in (1, 2, 4, 8):
        # Warm the prompt cache and settle the allocator before measuring.
        warm = GenerationJob(CHUNK, REF_AUDIO, REF_TEXT, None, 0.9, 16, False, False)
        scheduler.submit(warm)
        warm.result(timeout=600)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        jobs = [
            GenerationJob(CHUNK, REF_AUDIO, REF_TEXT, None, 0.9, 16, False, False)
            for _ in range(batch_size)
        ]
        start = time.perf_counter()
        for job in jobs:
            scheduler.submit(job)
        for job in jobs:
            job.result(timeout=600)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start

        per_item = wall / batch_size
        baseline = baseline or per_item
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"{batch_size:>6} {wall:>8.2f} {per_item:>8.2f} "
            f"{baseline / per_item:>7.2f}x {peak_gb:>8.2f}"
        )

    scheduler.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark and record the numbers**

Check the GPU is free first: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`

Run: `venv/bin/python bench/bench_batch.py`
Expected: a four-row table. Record the actual speedup and peak VRAM in `TESTING.md` under "Measured results", replacing the estimates. **If speedup at batch 4 is below 1.5x, stop and reconsider** — the scheduler is not paying for its complexity.

- [ ] **Step 3: Write the GPU concurrency test**

Create `tests/test_streaming_integration.py`:

```python
from __future__ import annotations

import concurrent.futures

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gpu

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "The river runs slowly behind the distant mountains."
)


@pytest.fixture(scope="module")
def client():
    import streaming_api_omnivoice as api

    with TestClient(api.app) as test_client:
        yield test_client


def fetch(client, voice_id):
    response = client.post(
        "/api/stream-mp3",
        json={"text": TEXT, "voice_id": voice_id, "chunk_chars": 120},
    )
    assert response.status_code == 200, response.text
    return response.content


def test_four_concurrent_streams_all_return_playable_audio(client):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        payloads = list(pool.map(lambda _: fetch(client, "af_heart"), range(4)))

    for payload in payloads:
        assert len(payload) > 1000, "stream returned suspiciously little audio"
        # MP3 frames start with a sync word (11 set bits) or an ID3 tag.
        assert payload[:3] == b"ID3" or payload[0] == 0xFF


def test_different_voices_batch_together_and_differ(client):
    import voices as voice_module

    ids = list(voice_module.VOICE_METADATA)[:2]
    if len(ids) < 2:
        pytest.skip("need at least two configured voices")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda v: fetch(client, v), ids))

    assert first != second, "distinct voices produced identical audio"
```

- [ ] **Step 4: Run the GPU tests**

Run: `venv/bin/python -m pytest tests/test_streaming_integration.py -m gpu -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Capture the "after" measurement and compare against Task 0**

Restart the API so it picks up the new code, then run the same harness:

```bash
venv/bin/uvicorn streaming_api_omnivoice:app --host 0.0.0.0 --port 9000   # terminal 1
venv/bin/python bench/bench_concurrent.py --label after                   # terminal 2
venv/bin/python bench/bench_concurrent.py --compare before after
```

Expected: `succeeded 8/8`, and `speedup_vs_serial` risen from ~1.0x to somewhere
around 2.5-3.5x. Paste the comparison table into `TESTING.md` under
"Measured results".

Read three things, not one:

- **`speedup_vs_serial`** — the headline. Below 1.5x means the scheduler is not
  earning its complexity; stop and investigate rather than shipping it.
- **`ttfb median`** — should stay close to the baseline. A large rise means the
  10 ms collection window or batch padding is hurting first-byte latency, which
  is what users actually feel on a streaming endpoint.
- **`total max`** — the slowest of the 8. Should fall sharply; under the old
  lock the 8th request waited for all seven ahead of it.

A run where `speedup_vs_serial` improves but `ttfb median` regresses badly is
not a win — it trades the metric users notice for one they do not.

- [ ] **Step 6: Commit**

```bash
git add bench/bench_batch.py bench/results/after.json tests/test_streaming_integration.py TESTING.md
git commit -m "test: add batch benchmark and concurrent streaming integration tests"
```

---

## Open decisions carried into this plan

Settled by me rather than discussed — the parts most worth pushing back on:

1. **`max_batch = 4`.** Matches the 2-4 concurrent-stream target. Because classifier-free guidance allocates every tensor at `2 * B`, batch 4 is 8 rows through the transformer. Raise only after reading the benchmark's VRAM column.
2. **10 ms collection window.** Long enough for near-simultaneous arrivals to meet, short enough to be invisible next to a multi-hundred-millisecond generation. A single stream pays 10 ms per chunk for nothing — if that shows up in the benchmark, skip the wait when the queue is empty and only one job is pending.
3. **No length bucketing.** Inputs pad to `max_c_len` and `max(target_lens)`, so mixing a 40-char chunk with a 600-char one wastes work. Deferred as YAGNI: `chunk_chars` already bounds the spread. Revisit if the benchmark shows batch efficiency collapsing on mixed sizes.
4. **Deferred jobs go to the queue tail.** A steady stream of one `batch_key` could in principle starve another. Harmless while all traffic shares the default config; a fairness fix would need per-key queues.
5. **Synchronous generator, not async.** A stream holds one anyio threadpool thread while waiting on its chunk; at `max_streams = 16` against the 40-thread default there is headroom. Switch to an `async` generator awaiting an `asyncio.Future` if `max_streams` ever approaches ~30.
6. **One in-flight chunk per stream.** Preserves ordering with no reorder buffer, at the cost of never pipelining chunk N+1's generation against chunk N's encoding.
