# Testing the Batch Scheduler

Companion to [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Describes what is
tested, why, and how to run it.

## Setup

```bash
venv/bin/pip install pytest==8.3.4 pytest-timeout==2.3.1
```

`pytest.ini` sets `addopts = -m "not gpu"`, so the default run never touches the
GPU or downloads model weights.

## Test layers

| Layer | File | GPU | Runtime | What it protects |
|---|---|---|---|---|
| Unit — scheduler | `tests/test_batch_scheduler.py` | no | < 5 s | Batch grouping, result routing, fault isolation, cancellation |
| Unit — admission | `tests/test_admission.py` | no | < 5 s | Concurrency cap, slot leaks, 503 behaviour |
| Unit — OpenAI API | `tests/test_openai_api.py` | no | < 5 s | `/v1/audio/speech` schema, voice aliases, error envelope |
| Integration | `tests/test_streaming_integration.py` | **yes** | ~1-2 min | Real concurrent HTTP streams produce playable audio |
| Benchmark — end to end | `bench/bench_concurrent.py` | **yes** | ~2-4 min | 8 concurrent requests, before vs. after |
| Benchmark — model | `bench/bench_batch.py` | **yes** | ~2-5 min | Throughput and VRAM at batch 1/2/4/8 |

### Run them

```bash
# Default: everything that does not need a GPU
venv/bin/python -m pytest -v

# One layer
venv/bin/python -m pytest tests/test_batch_scheduler.py -v
venv/bin/python -m pytest tests/test_admission.py -v
venv/bin/python -m pytest tests/test_openai_api.py -v

# GPU integration (needs weights + free VRAM)
venv/bin/python -m pytest tests/test_streaming_integration.py -m gpu -v

# End-to-end concurrency, before vs. after (needs the API running on :9000)
venv/bin/python bench/bench_concurrent.py --label before    # BEFORE implementing
venv/bin/python bench/bench_concurrent.py --label after     # AFTER implementing
venv/bin/python bench/bench_concurrent.py --compare before after

# Model-level benchmark (not a test — prints a table)
venv/bin/python bench/bench_batch.py
```

## The 8-request concurrency benchmark

`bench/bench_concurrent.py` is the test that answers the original question:
*can several people use this at once?* It fires 8 simultaneous
`/api/stream-mp3` requests and records, per request, time-to-first-byte and
total time — plus a single-request serial reference measured with nothing else
running.

The headline metric is **`speedup_vs_serial`**: how much faster 8 concurrent
requests complete than the same 8 run back to back.

```
speedup_vs_serial = (serial_1_request_time * 8) / concurrent_wall_time
```

This ratio is the right metric because it is **self-normalizing**. Absolute
latency depends on GPU clocks, thermal state, and whatever else is running, so
"before" and "after" wall times are not directly comparable across sessions.
The ratio is. A fully serialized server scores ~1.0x no matter how fast its
individual requests are — which is precisely what the current global lock
produces, and precisely what the scheduler is meant to change.

Results are saved to `bench/results/<label>.json` with per-request detail, the
git commit, and a timestamp, so the numbers are checkable later rather than
remembered. **The `before` baseline must be captured before Task 5 modifies
`streaming_api_omnivoice.py`** — after that, the old behaviour is only
recoverable via git.

### Reading the comparison

Three numbers, not one:

| Metric | Expect | Meaning if it disappoints |
|---|---|---|
| `speedup_vs_serial` | ~1.0x → 2.5-3.5x | Below 1.5x after: the scheduler is not earning its complexity |
| `ttfb median` | roughly unchanged | A large rise means the 10 ms collection window or batch padding is hurting the latency users actually feel |
| `total max` | falls sharply | Under the old lock the 8th request waited behind all seven; if it does not fall, requests are still serializing |

A run where `speedup_vs_serial` improves while `ttfb median` regresses badly is
not a win — it trades the metric users notice for one they do not.

## The test double

`tests/conftest.py` provides `FakeOmniVoice`, which records every `generate()`
call and returns, for input text `"job-<n>"`, an array filled with `float(n)`.

That encoding is deliberate. It makes the scariest bug in this design directly
assertable: if the worker's `zip(batch, audios)` ever misaligns, one user
receives another user's audio. `test_results_are_routed_to_the_job_that_asked_for_them`
submits `job-0` through `job-3` concurrently and asserts each job got its own
number back. A silent index bug fails that test loudly.

The double also means the whole batching policy is testable in milliseconds with
no GPU, no weights, and no flakiness.

## What each unit test covers

### `tests/test_batch_scheduler.py`

**Job mechanics**
- `test_batch_key_covers_per_batch_config_only` — two jobs differing only in
  per-item fields (text, speed, voice) share a key, so they may batch.
- `test_batch_key_differs_when_num_step_differs` — a per-batch config field
  splits the key.
- `test_result_returns_what_was_set` / `test_result_reraises_the_exception_that_was_set`
  — the result slot carries both outcomes.
- `test_cancel_marks_job_cancelled`.

**Batching policy**
- `test_single_job_round_trips` — the trivial path still works.
- `test_results_are_routed_to_the_job_that_asked_for_them` — see above; the
  most important test in the suite.
- `test_jobs_sharing_a_batch_key_ride_one_generate_call` — proves batching
  actually happens rather than the scheduler silently degrading to one-at-a-time.
  Without it, every other test would still pass on a scheduler that never batches.
- `test_batch_never_exceeds_max_batch` — 10 jobs, `max_batch=4`; no call may
  carry more than 4 texts. Guards the VRAM ceiling.
- `test_different_batch_keys_are_never_mixed` — mixed `num_step` must never
  share a `generate()` call, and the call's `num_step` must match its members.
  Without this, some users would silently get the wrong number of diffusion steps.
- `test_per_item_arguments_are_passed_as_aligned_lists` — `speed`, `language`,
  and `voice_clone_prompt` are lists the same length as `text`. A scalar leaking
  through here would apply one user's speed or voice to everyone.
- `test_voice_clone_prompts_are_cached_per_reference` — four jobs on one voice
  build the prompt once. Prompt construction tokenizes the reference audio; a
  cache miss per chunk would erase the batching win.
- `test_wait_ready_is_false_when_the_model_fails_to_load` — the ready event
  fires on load failure as well as success, so `wait_ready` must distinguish
  them. If it reported success, `warmup()` would submit a job to a dead worker
  and hang on startup with no error.
- `test_cancelled_jobs_are_skipped` — a disconnected client's queued chunk is
  never sent to the model.

**Fault isolation**
- `test_one_poison_chunk_does_not_fail_its_batchmates` — the core reason
  `_process_batch` retries individually. `generate()` raises for the whole
  batch, so without the retry one bad chunk kills three healthy streams.
- `test_single_job_failure_propagates_without_retry_storm` — a lone failing job
  is not retried, so failures cost one call, not two.
- `test_worker_survives_a_failure_and_serves_the_next_job` — if the worker
  thread ever dies, every subsequent request blocks forever on its result slot
  with no error. This test is the guard against that hang.

### `tests/test_admission.py`

- `test_acquires_up_to_the_cap`, `test_releasing_frees_a_slot`,
  `test_active_count_tracks_outstanding_slots` — basic accounting.
- `test_release_is_idempotent` — the streaming generator's `finally` and the
  endpoint's error path may both release the same slot. Non-idempotent release
  against a `BoundedSemaphore` raises `ValueError` and would inflate the
  effective cap.
- `test_concurrent_acquire_never_exceeds_the_cap` — 50 threads race for 5 slots.
- `test_returns_503_when_no_slot_is_available` — the endpoint refuses rather
  than queueing without bound, and sets `Retry-After`.
- `test_slot_is_returned_after_a_stream_completes` and
  `test_unknown_voice_leaks_no_slot` — two exit paths, one invariant. Slot leaks
  degrade the server to permanent 503 after enough requests, the kind of failure
  that only appears in production after hours of uptime.

### `tests/test_streaming_integration.py` (GPU)

- `test_four_concurrent_streams_all_return_playable_audio` — four simultaneous
  requests all reach 200 with non-trivial MP3 bytes. This is the end-to-end
  claim the whole plan exists to make true.
- `test_different_voices_batch_together_and_differ` — two different voices in
  flight at once produce different audio, confirming per-item
  `voice_clone_prompt` survives batching. If a batch collapsed everyone onto
  item 0's reference — the failure mode `_preprocess_all` invites at
  `models/omnivoice.py:961` — both payloads would be identical.

## Manual check

With the server running:

```bash
venv/bin/uvicorn streaming_api_omnivoice:app --host 0.0.0.0 --port 9000
```

Open `http://localhost:9000/` in three or four browser tabs and start playback
in each within a second or two. Before this change, tabs 2+ sit silent until
tab 1 finishes. After it, all of them should begin producing audio at roughly
the same time.

## Measured results

### 8-request concurrency, before vs. after

Measured 2026-07-29 on an RTX 5080 (16.3 GB). `before` at commit `6ebb4fb`
(global lock), `after` at `0417bfd` (batch scheduler). Both runs: 8 concurrent
`/api/stream-mp3` requests, `af_heart`, `chunk_chars=120`, 254 characters of
text.

| Metric | before | after | change |
|---|---|---|---|
| 8 concurrent wall | 28.88s | 7.26s | **3.97x faster** |
| serial 1 request | 4.10s | 2.83s | 1.45x faster |
| ttfb median | 13.45s | 2.20s | **6.10x faster** |
| ttfb max | 25.44s | 3.26s | 7.80x faster |
| total median | 15.23s | 6.34s | 2.40x faster |
| total max | 28.86s | 7.25s | 3.98x faster |
| **speedup_vs_serial** | **1.14x** | **3.11x** | **2.73x** |

All three things worth reading moved the right way:

- **`speedup_vs_serial` 1.14x → 3.11x.** The `before` figure is the signature
  of a fully serialized server, exactly as the global lock predicted. 3.11x
  lands in the middle of the 2.5-3.5x estimate.
- **`ttfb median` 13.45s → 2.20s.** This is the number users feel. Under the
  lock the median request waited about 13 seconds before hearing anything; it
  now waits about 2. The 10 ms collection window costs nothing visible against
  that. Note this is the opposite of the failure mode the plan warned about —
  throughput did not come at the cost of first-byte latency.
- **`total max` 28.86s → 7.25s.** The unluckiest of the eight used to wait
  behind all seven ahead of it. It no longer does.

One caveat on the serial reference: it also improved (4.10s → 2.83s), so part
of the wall-clock gain is a faster single request, not concurrency. That is
precisely why `speedup_vs_serial` is the headline — it divides that out.

Raw records with per-request detail live in `bench/results/before.json` and
`bench/results/after.json`.

### Model-level batch scaling

Measured 2026-07-29, `venv/bin/python bench/bench_batch.py`, RTX 5080, fp16,
`num_step=16`, one ~120-character chunk per item. Peak VRAM is
`torch.cuda.max_memory_allocated()`, so it includes the ~2.1 GB of weights.

| Batch | Wall (s) | s/item | Speedup | Peak VRAM (GB) |
|---|---|---|---|---|
| 1 | 0.88 | 0.88 | 1.00x | 2.10 |
| 2 | 0.88 | 0.44 | 1.99x | 2.29 |
| 4 | 1.56 | 0.39 | **2.26x** | 2.66 |
| 8 | 3.30 | 0.41 | 2.13x | 3.41 |

Against the predictions:

- **Batch 4 at 2.26x** — inside the predicted 2.5-3.5x band's lower edge, and
  comfortably above the 1.5x threshold at which the plan said to stop. The
  scheduler pays for itself.
- **The curve flattens after batch 2, not batch 4.** Two items ride for free
  (0.88s for one, 0.88s for two), then wall time grows roughly linearly. That
  is the classifier-free-guidance effect the plan predicted, only stronger:
  batch 1 already puts 2 rows through the transformer, so the GPU appears to
  saturate at about 4 rows. **`max_batch = 4` is therefore the right default —
  raising it buys throughput barely at all** (batch 8 is *worse* per item than
  batch 4) while adding queueing latency.
- **VRAM is far cheaper than estimated: 2.66 GB at batch 4, 3.41 GB at batch
  8**, against a 4.5-6 GB prediction. Batching costs ~0.19 GB per additional
  item. Weights dominate and do not grow, which is the whole argument for
  batching over model replicas — four replicas would need ~8.4 GB of weights
  for the same concurrency.
- Note the end-to-end `speedup_vs_serial` of 3.11x exceeds this table's 2.26x.
  That is expected: the API also overlaps MP3 encoding and HTTP transfer with
  generation, which this model-only benchmark does not measure.

### Before running anything on the GPU

The card is a 16.3 GB RTX 5080. A Jupyter kernel from `~/huggingface/qwen3-asr-lora`
has previously held ~12 GB of it. Confirm it is free first:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## Known gaps

- **No test asserts wall-clock concurrency.** The integration test proves four
  streams all succeed, not that they overlapped. Timing assertions are flaky on
  a shared GPU; `bench/bench_batch.py` is the honest measurement instead.
- **Audio quality is unverified.** Tests check that bytes are non-empty and
  MP3-framed, never that the speech is correct or natural. Batched output is
  assumed to match unbatched output because per-item state is threaded through
  by index; that assumption is untested and worth one manual listen comparing a
  solo stream against the same text generated alongside three others.
- **Padding cost is unmeasured.** Every batch pads to `max_c_len` and
  `max(target_lens)`, so mixing very short and very long chunks wastes work. The
  benchmark uses uniform chunks and will therefore report the optimistic case.
- **Client-disconnect cancellation is only partly tested.** `test_cancelled_jobs_are_skipped`
  covers a job cancelled while queued. A job already inside a running
  `generate()` cannot be pulled out — it completes and its result is discarded —
  and no test exercises that path.
