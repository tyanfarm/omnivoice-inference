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


class JobCancelled(Exception):
    """Raised by GenerationJob.result() when the job was cancelled first.

    The scheduler drops a cancelled job instead of generating it, so nobody
    would otherwise fill its result slot and the caller would block for the
    whole timeout holding a worker thread.
    """


@dataclass
class GenerationJob:
    """One text chunk awaiting generation.

    cancel(), set_result(), and set_exception() are three possible writers to
    the job's one-item result slot; whichever writes first wins and the other
    writes are silently dropped. The submitting thread blocks in result()
    until the slot is filled, then either returns audio or raises the
    delivered exception — JobCancelled if cancel() won the race.
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
        """Mark the job dead and release anyone blocked in result().

        Idempotent: the flag is set unconditionally, and re-delivering is
        harmless either way — a full slot drops the payload, an empty one
        (already drained by a prior consumer) just delivers a JobCancelled
        nobody will ever read.
        """
        self._cancelled.set()
        self._deliver((None, JobCancelled("generation cancelled")))

    def set_result(self, audio: np.ndarray) -> None:
        self._deliver((audio, None))

    def set_exception(self, exc: BaseException) -> None:
        self._deliver((None, exc))

    def _deliver(self, payload: tuple[np.ndarray | None, BaseException | None]) -> None:
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

    def result(self, timeout: float | None = None) -> np.ndarray:
        audio, exc = self._slot.get(timeout=timeout)
        if exc is not None:
            raise exc
        return audio


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
