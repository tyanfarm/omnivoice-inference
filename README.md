```sh
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Set environment variable for timeout
export UV_HTTP_TIMEOUT=300

# Install PyTorch with CUDA 12.8 support
uv pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install omnivoice

uv pip install "lameenc==1.8.2" "websockets==16.1.1"

uvicorn streaming_api_omnivoice:app --host 0.0.0.0 --port 9000
```

## OpenAI-compatible endpoint

`POST /v1/audio/speech` streams audio using OpenAI's path and `input` field. Same
engine, admission control, and batching as `/api/stream-mp3`.

```sh
curl -X POST http://localhost:9000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello there.","voice":"af_heart"}' \
  --output speech.mp3
```

Three fields:

- **`input`** — the text (required, no length limit; it is split into chunks and
  streamed as each one finishes).
- **`voice`** — `af_heart`, `am_michael`, `vf_phuong`, `vm_thanh`, `vf_quynh`
  (see `GET /api/voices`). Defaults to `vf_phuong`. An unknown voice is a 400
  rather than a silent substitution with some other speaker.
- **`response_format`** — `mp3` (default), `wav`, or `pcm`. `opus`, `aac`, and
  `flac` are not produced; asking for one is a 400 that names the formats that
  work. `wav` is a 44-byte RIFF header followed by the same bytes `pcm` returns:
  16-bit signed little-endian mono at the model's 24 kHz. Because the length is
  unknown while streaming, the header's size fields carry `0xFFFFFFFF` and
  players read to end-of-stream.

Everything else — speed, chunking, diffusion steps — uses the same defaults as
`/api/stream-mp3`. Extra fields a stock OpenAI client sends (`model`, `speed`,
`instructions`) are ignored rather than rejected, so those clients still work.
No `Authorization` header is required; one sent is ignored.

Errors use OpenAI's envelope:

```json
{"error":{"message":"Voice not found: nope","type":"invalid_request_error","param":"voice","code":null}}
```

## WebSocket streaming

`WS /v1/audio/speech/ws?voice=vf_phuong&response_format=pcm` streams text in and
audio out on one connection. This is **not** an OpenAI endpoint — OpenAI has no
WebSocket TTS — so the protocol is specific to this server.

Two query parameters, and no others: `voice` (default `vf_phuong`) and
`response_format` (`pcm` default, `wav`, `mp3`). Speed and diffusion steps use
the same defaults as `/v1/audio/speech`; chunk size is 240 characters here
against that endpoint's 120, so the same text can be cut at different chunk
boundaries.

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

`uvicorn` is pinned without `[standard]`, so the `websockets` package in
`requirements.txt` is what makes the handshake work. Without it the upgrade is
refused.

## Note: vLLM-Omni cannot serve OmniVoice concurrently

Investigated 2026-07-29 and rejected. Recording it here so nobody repeats it.

vLLM-Omni does support `k2-fsa/OmniVoice` (`vllm serve k2-fsa/OmniVoice --omni`),
and it starts and generates audio fine. But it serves **one request at a time**.
There is no way to enable request-level batching, so it offers no concurrency
advantage over running the model in-process — while adding a second process, an
HTTP hop per chunk, and a ~12 GB virtualenv.

`vllm_omni/diffusion/diffusion_engine.py` resolves exactly two execution modes,
and OmniVoice is locked out of both:

- **`REQUEST_BATCH`** requires the pipeline class to declare
  `supports_request_batch = True`. `sd3`, `flux`, `qwen_image`, and `ltx2` do.
  OmniVoice never declares it, so it defaults to `False` and the engine refuses
  to start with `max_num_seqs > 1`:

  ```
  ValueError: 'OmniVoice' does not support request-level batching.
  Use max_num_seqs=1 for serial request execution, or choose a pipeline
  with supports_request_batch=True.
  ```

- **`STEP_BATCH`** skips that check, but requires the pipeline to implement the
  `SupportsStepExecution` protocol. `OmniVoiceModel` implements no step methods,
  and no shipped pipeline implements that protocol at all. This is also why
  OmniVoice is marked non-streaming in vLLM-Omni's model table — streaming
  requires `step_execution=True`, which requires the same missing protocol.

The irony: the standalone `omnivoice` package **does** batch natively.
`OmniVoice.generate()` accepts `text` as a list with per-item
`voice_clone_prompt`, `speed`, and `language` (see also its `cli/infer_batch.py`).
The capability exists — it just isn't reachable through vLLM-Omni yet. Hence the
in-process batch scheduler in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

Worth revisiting only if vllm-omni adds `supports_request_batch` to the OmniVoice
pipeline.

### Incidental findings, if you do revisit

- **vLLM runs fine on consumer Blackwell (RTX 5080, sm_120).** `torch.cuda.get_arch_list()`
  includes `sm_120`, capability reports `(12, 0)`, and fp16 kernels dispatch.
- **Use the cu129 wheel.** The default PyPI `vllm` wheel is a CUDA 13 build.
  Paired with `--torch-backend=auto` (which resolves to cu128 torch on uv 0.8.0)
  it dies with `ImportError: libcudart.so.13`. v0.26.0 publishes only `cu129`
  and the default cu13 build — there is no cu128 or cu130 wheel.

  ```sh
  uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      --extra-index-url https://download.pytorch.org/whl/cu129
  ```

- **Version-match `vllm` and `vllm-omni`.** `vllm-omni 0.24.0` against `vllm 0.26.0`
  fails with `ModuleNotFoundError: No module named 'vllm.entrypoints.serve.disagg'`.
  vllm-omni's stable releases lag vLLM's, so tracking vLLM means using release
  candidates (`vllm-omni==0.26.0rc1`). Plain `pip install vllm vllm-omni` yields a
  broken pair.
- **Keep it in a separate virtualenv.** vLLM pins its own `torch` (2.11.0) and
  will replace this project's `torch 2.8.0+cu128`.
