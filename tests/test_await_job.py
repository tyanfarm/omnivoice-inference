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
