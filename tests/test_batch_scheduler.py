from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from batch_scheduler import BatchScheduler, GenerationJob, JobCancelled


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
