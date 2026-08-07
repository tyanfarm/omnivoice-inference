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

## Torch-native inference: onnx2torch + a graph patch

Loading path (runs once at service startup, mirrors
`OmniVoiceStreamingService._load_model`):

1. Download `smart-turn-v3.2-gpu.onnx` from the HF Hub (cached locally after
   first run).
2. Patch the graph in memory: 16 `Reshape` nodes in this export carry
   `allowzero=1`, which the installed `onnx2torch` (1.5.15) does not
   support (`NotImplementedError`). Flip the attribute to `0` on those
   nodes. This is safe here — verified against the actual graph that none
   of their target shapes are ever literally `0` (they're attention-head
   reshapes with fixed positive dims for this fixed 8s/16kHz input) — so
   `allowzero=0` and `allowzero=1` are behaviorally identical for this
   model.
3. `onnx2torch.convert(...)` the patched graph into a `torch.nn.Module`,
   `.eval()`, move to `cuda:0` (or CPU if no GPU).

Verified: the converted torch module's output matches ONNX Runtime's output
on the original (unpatched) graph to ~1e-7 absolute difference across
multiple random inputs, and both CPU and GPU execution paths work. GPU
inference measured at ~18ms, consistent with upstream's published benchmarks.

No onnxruntime dependency is needed at serving time — only `onnx` (to load
and patch the graph) and `onnx2torch` (to build the torch module).
`onnxruntime` is added as a dev/test-only dependency, used exclusively by a
regression test that re-checks the patched conversion still matches ONNX
Runtime's reference output (guards against a future onnx2torch/onnx upgrade
silently breaking the patch).

## New module: `turn_detection.py`

Mirrors the existing `OmniVoiceStreamingService` pattern:

- `_patch_reshape_allowzero(onnx_model) -> onnx.ModelProto` — the graph
  patch described above.
- `_load_torch_model(onnx_path, device) -> torch.nn.Module` — download,
  patch, convert, move to device, eval mode.
- `_truncate_audio_to_last_n_seconds(audio, n_seconds=8) -> np.ndarray` —
  vendored (BSD-2-Clause, credited in a comment) from smart-turn's
  `audio_utils.py`: keeps the *end* of the clip and pads at the *start* if
  shorter, matching upstream's own preprocessing (most recent audio is the
  most relevant to a turn-completion decision).
- `TurnDetectionService`:
  - `warmup()` — loads the model, called from FastAPI's startup event like
    `service.warmup()` today.
  - `predict(audio: np.ndarray) -> dict` — truncate/pad → 
    `WhisperFeatureExtractor` → forward pass → 
    `{"prediction": 0 | 1, "probability": float}` (sigmoid is already
    applied inside the graph, so no extra activation step is needed;
    `prediction = 1 if probability > 0.5 else 0`).

## Audio input

`POST /v1/turn/predict` accepts **multipart form-data**, field name `file`
— matching the calling convention of the user's existing STT service
(`curl --form 'file=@clip.mp3' .../v1/audio/transcriptions`), so the same
client code / pattern can hit both endpoints. Any container ffmpeg
understands (mp3, wav, m4a, ...) is accepted.

Decoding uses `torchaudio.load` (already installed and used elsewhere in
this repo — confirmed it decodes this repo's own `.mp3` voice samples via
its ffmpeg backend, so no new dependency is needed for decoding):

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

- `turn_service = TurnDetectionService()` at module scope in
  `streaming_api_omnivoice.py` (cheap — no model load in `__init__`).
- Model load happens in the existing `@app.on_event("startup")` handler,
  alongside `service.warmup()`.
- `/health` gains a `turn_model_loaded` boolean next to the existing
  `model_loaded`.
- No admission control / batch scheduler for this endpoint: unlike
  OmniVoice's diffusion generation, this is a single ~18ms forward pass, so
  it runs directly in FastAPI's threadpool like the other synchronous `def`
  routes. Revisit if concurrent load turns out to be heavy enough to
  contend for GPU time with OmniVoice generation.

## Dependencies (`requirements.txt`)

- `onnx` — runtime, used to load/patch the ONNX graph at startup.
- `onnx2torch` — runtime, converts the patched graph to a torch module.
- `onnxruntime` — dev/test only (alongside the existing pytest comment
  block), used by the conversion-parity regression test.

`torch`/`torchaudio`/`transformers` are already present in this venv (per
the README's manual install step and `omnivoice`'s own dependency chain
respectively) — no changes needed there.

## Testing

New `tests/test_turn_detection.py`, following the existing test file
conventions in `tests/`:

- Audio decode helpers (mono downmix, resampling, last-N-seconds
  truncate/pad) — pure functions, no model needed.
- Conversion parity: build the patched torch module and assert its output
  matches an ONNX Runtime session on the same (unpatched) graph within a
  tight tolerance, on a couple of fixed inputs — this is the regression
  guard for the `allowzero` patch specifically.
- `predict()` integration test against the real downloaded model, gated the
  same way `test_streaming_integration.py` gates its real-model tests
  (slow/optional).
