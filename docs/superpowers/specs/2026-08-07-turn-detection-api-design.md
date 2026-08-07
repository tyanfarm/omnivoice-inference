# Turn detection API — design

## Goal

Add `POST /v1/turn/predict` to `streaming_api_omnivoice.py`: given an audio
clip of a user's current speaking turn, return whether the turn is
semantically complete (the speaker is done talking) or incomplete, using
[pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) v3.

This runs alongside the existing OmniVoice TTS service in the same process,
sharing the same GPU and the same torch/transformers install already present
in this venv.

## Model

Official weights: `pipecat-ai/smart-turn-v3` on Hugging Face,
`smart-turn-v3.2-gpu.onnx` (fp32 variant — matches running on the same GPU
as OmniVoice). Architecture: Whisper-tiny encoder (4 layers) + attention
pooling + a small linear classifier head, ~8M params. Input: log-mel
features of up to 8 seconds of 16kHz mono audio (`WhisperFeatureExtractor`,
`chunk_length=8` → shape `(80, 800)`). Output: a single sigmoid probability
that the turn is complete.

This is the only smart-turn release with a from-scratch PyTorch
reimplementation documented in the upstream repo's `train.py`
(`SmartTurnV3Model`), but the *published checkpoint* for v3 is ONNX-only —
there is no safetensors/PyTorch checkpoint on the Hub for v3, unlike v2.

## Inference: onnxruntime-gpu, the same runtime upstream ships

Earlier iterations of this design (see git history) tried a torch-native
route: convert the ONNX graph to a `torch.nn.Module` with `onnx2torch` so
inference never leaves torch. That conversion worked (verified to ~1e-7
parity with ONNX Runtime after patching one unsupported `Reshape` attribute)
but traded a real, if small, risk — a hand-patched graph, and a conversion
library whose op coverage isn't guaranteed across upgrades — for a benefit
that turned out not to matter in practice: upstream's own `inference.py`
already runs this exact checkpoint via `onnxruntime-gpu` with
`CUDAExecutionProvider`, and that path is faster (~3.4ms/inference measured
here vs ~18ms for the onnx2torch route) with zero patching. Given
`onnxruntime-gpu` shares the GPU via CUDA the same way torch does — a
separate CUDA context, but the same device — the extra runtime was judged
worth it for matching upstream exactly. The design now uses that path.

Loading path (runs once at service startup, mirrors
`OmniVoiceStreamingService._load_model`):

1. Download `smart-turn-v3.2-gpu.onnx` from the HF Hub (cached locally after
   first run) via `huggingface_hub.hf_hub_download`.
2. Build an `onnxruntime.InferenceSession` over it, with
   `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` when
   `torch.cuda.is_available()` (torch is only consulted here for the device
   check — OmniVoice already depends on it), else CPU-only.

Verified: `session.get_providers()` confirms `CUDAExecutionProvider` is
actually selected (not silently falling back to CPU) in this venv, and a
real end-to-end request against a running server returns a sane probability
in well under 100ms.

`onnxruntime-gpu` is a runtime dependency (not dev/test-only, unlike the
earlier onnx2torch draft) — it's the actual inference engine now. No `onnx`
package or graph patching is needed.

## `turn_detection.py`

Owns everything about turn detection — model loading, audio decoding, and
the service singleton — so `streaming_api_omnivoice.py` stays limited to
route definitions:

- `_load_onnx_session() -> onnxruntime.InferenceSession` — download, pick
  providers, build the session.
- `_truncate_audio_to_last_n_seconds(audio, n_seconds=8) -> np.ndarray` —
  vendored (BSD-2-Clause, credited in a comment) from smart-turn's
  `audio_utils.py`: keeps the *end* of the clip and pads at the *start* if
  shorter, matching upstream's own preprocessing (most recent audio is the
  most relevant to a turn-completion decision).
- `_downmix_to_mono`, `_resample_if_needed`, `decode_audio_upload(data: bytes) -> np.ndarray`
  — the audio-decode pipeline described below (moved here from the API
  module so the API module has no audio-processing logic of its own).
- `TurnDetectionService`:
  - `warmup()` — loads the model, called from FastAPI's startup event like
    `service.warmup()` today.
  - `is_ready: bool` — property, used by `/health`.
  - `predict(audio: np.ndarray) -> dict` — truncate/pad →
    `WhisperFeatureExtractor` → `session.run(...)` →
    `{"prediction": 0 | 1, "probability": float}` (sigmoid is already
    applied inside the graph, so no extra activation step is needed;
    `prediction = 1 if probability > 0.5 else 0`).
- `turn_service = TurnDetectionService()` — the module-level singleton,
  constructed here (not in the API module) and imported by
  `streaming_api_omnivoice.py`.

## Audio input

`POST /v1/turn/predict` accepts **multipart form-data**, field name `file`
— matching the calling convention of the user's existing STT service
(`curl --form 'file=@clip.mp3' .../v1/audio/transcriptions`), so the same
client code / pattern can hit both endpoints. Any container ffmpeg
understands (mp3, wav, m4a, ...) is accepted.

Decoding uses `torchaudio.load` (already installed and used elsewhere in
this repo — confirmed it decodes this repo's own `.mp3` voice samples via
its ffmpeg backend, so no new dependency is needed for decoding). All of
this lives in `turn_detection.decode_audio_upload`, not the API module:

1. `torchaudio.load(BytesIO(upload_bytes))` → `(channels, samples)` float32
   tensor + sample rate.
2. Downmix to mono if multi-channel (`mean(dim=0)`).
3. `torchaudio.functional.resample` to 16kHz if the source sample rate
   differs.
4. `.numpy()` → into `TurnDetectionService.predict`.

Errors (empty file, undecodable audio, oversized upload) return a plain
`400 HTTPException` — this endpoint is not OpenAI-shaped, so it follows the
WS endpoint's own (non-OpenAI-envelope) error convention rather than
`/v1/audio/speech`'s OpenAI envelope. A soft upload-size cap (25MB, matching
Whisper's own limit) rejects obviously-wrong uploads before decode is
attempted.

## Endpoint

```python
@app.post("/v1/turn/predict")
def predict_turn(file: UploadFile = File(...)) -> dict:
    ...
```

Response: `{"prediction": 1, "probability": 0.87}` — kept identical to
smart-turn's own dict shape (no field renaming) since it's a direct
pass-through of upstream's semantics.

## Wiring

- `turn_service = TurnDetectionService()` is a module-level singleton in
  `turn_detection.py` itself (cheap — no model load in `__init__`), and
  `streaming_api_omnivoice.py` imports it (`from turn_detection import
  decode_audio_upload, turn_service`) rather than constructing it. This
  keeps the API module limited to route definitions, with no
  audio-processing or model-loading logic of its own — everything turn-
  detection-specific lives in `turn_detection.py`.
- Model load happens in the existing `@app.on_event("startup")` handler,
  alongside `service.warmup()`.
- `/health` gains a `turn_model_loaded` boolean (`turn_service.is_ready`)
  next to the existing `model_loaded`.
- No admission control / batch scheduler for this endpoint: unlike
  OmniVoice's diffusion generation, this is a single sub-10ms forward pass
  on GPU, so it runs directly in FastAPI's threadpool like the other
  synchronous `def` routes. Revisit if concurrent load turns out to be
  heavy enough to contend for GPU time with OmniVoice generation.

## Dependencies (`requirements.txt`)

- `onnxruntime-gpu` — runtime; this is the actual inference engine, not a
  dev/test-only tool. Pinned to `1.23.2`, matching the version upstream's
  own `requirements.txt` uses.
- `transformers`, `huggingface_hub` — already present transitively via
  `omnivoice`, pinned explicitly since `turn_detection.py` imports them
  directly.

`torch`/`torchaudio` are already present in this venv (per the README's
manual install step) — no changes needed there. `torch` is still used, but
only for `torch.cuda.is_available()` to pick onnxruntime's providers list;
it does no inference work for this model.

## Testing

`tests/test_turn_detection.py` (model/decode logic — mirrors where the code
now lives) and `tests/test_turn_endpoint.py` (HTTP-level behavior of the
route), following the existing test file conventions in `tests/`:

- Audio decode helpers (mono downmix, resampling, last-N-seconds
  truncate/pad) — pure functions, no model needed.
- `TurnDetectionService.predict()` logic against a stub `onnxruntime`-shaped
  session object (records the fed input, returns a fixed probability) —
  fast, no GPU/network.
- A real end-to-end `predict()` test against the downloaded model, gated
  the same way `test_streaming_integration.py` gates its real-model tests
  (the `gpu` pytest marker, excluded by default).
- HTTP-level endpoint tests (200 shape, empty file, undecodable audio,
  oversized upload, `/health` field) via the shared `client` fixture, which
  now also stubs `turn_service`'s session so app startup in tests never
  downloads the real model.
