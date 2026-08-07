# Turn Detection API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/turn/predict` to `streaming_api_omnivoice.py`, backed by smart-turn-v3 running as torch-native inference (via an onnx2torch conversion of the official ONNX checkpoint), so a client can upload an audio clip of a user's current speaking turn and get back whether it's semantically complete.

**Architecture:** A new `turn_detection.py` module owns the model: it downloads `pipecat-ai/smart-turn-v3`'s `smart-turn-v3.2-gpu.onnx`, patches one unsupported node attribute, converts it to a `torch.nn.Module` with `onnx2torch`, and exposes a `TurnDetectionService` with the same `warmup()`/state-holding shape as the existing `OmniVoiceStreamingService`. `streaming_api_omnivoice.py` gains audio-decode helpers (multipart upload → mono 16kHz numpy via `torchaudio`) and the endpoint itself, wired into the existing startup/health machinery.

**Tech Stack:** FastAPI, PyTorch 2.8 (`cuda:0`), `torchaudio` (already installed, ffmpeg backend), `transformers.WhisperFeatureExtractor`, `onnx` + `onnx2torch` (new), `onnxruntime` (dev/test only), `huggingface_hub`.

## Global Constraints

- Model: `pipecat-ai/smart-turn-v3`, file `smart-turn-v3.2-gpu.onnx` (fp32 GPU variant) — no other smart-turn version or file.
- Inference must be torch-native at serving time — no `onnxruntime` import outside tests.
- Audio input is multipart form-data, field name `file` (matches the user's existing STT service's calling convention), any container `torchaudio`'s ffmpeg backend can decode.
- Response shape is exactly `{"prediction": 0 | 1, "probability": <float>}` — no field renaming.
- No admission control / batch scheduler for this endpoint (single ~18ms forward pass).
- New runtime deps go in `requirements.txt` pinned to the exact versions already verified working in this venv: `onnx==1.22.0`, `onnx2torch==1.5.15`. `onnxruntime==1.23.2` is dev/test only. `transformers==5.9.0` and `huggingface_hub==1.16.4` get explicit pins too since `turn_detection.py` imports them directly (previously only transitive via `omnivoice`).
- Full design rationale, including the `allowzero` patch and its numerical-parity verification: `docs/superpowers/specs/2026-08-07-turn-detection-api-design.md`.

---

### Task 1: ONNX → torch model loader

**Files:**
- Modify: `requirements.txt`
- Modify: `pytest.ini:4` (the `gpu` marker's description)
- Create: `turn_detection.py`
- Create: `tests/test_turn_detection.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 3): `turn_detection.TURN_MODEL_REPO: str`, `turn_detection.TURN_MODEL_FILE: str`, `turn_detection._patch_reshape_allowzero(model: onnx.ModelProto) -> onnx.ModelProto`, `turn_detection._load_torch_model(device: torch.device) -> torch.nn.Module`.

- [ ] **Step 1: Add the new dependencies to `requirements.txt`**

Insert this block after the `websockets==16.1.1` line and before the `## Install pytorch...` comment:

```
# Turn detection (/v1/turn/predict): smart-turn-v3's published checkpoint is
# ONNX-only, so onnx2torch converts it to a torch.nn.Module once at startup
# so inference stays torch-native. transformers/huggingface_hub are already
# pulled in transitively by omnivoice, but are pinned explicitly here since
# turn_detection.py imports them directly.
onnx==1.22.0
onnx2torch==1.5.15
transformers==5.9.0
huggingface_hub==1.16.4
```

And append this line to the existing `## Dev / test only` comment block at the bottom of the file:

```
# pip install onnxruntime==1.23.2  # verifies the onnx2torch conversion still matches ONNX Runtime
```

- [ ] **Step 2: Install the new packages into the venv**

Run: `source venv/bin/activate && pip install onnx==1.22.0 onnx2torch==1.5.15 onnxruntime==1.23.2`
Expected: all three install successfully (transformers/huggingface_hub are already present in this venv at the pinned versions).

- [ ] **Step 3: Broaden the `gpu` pytest marker's description**

In `pytest.ini`, change:

```
    gpu: requires a CUDA GPU and downloaded OmniVoice weights
```

to:

```
    gpu: requires a CUDA GPU and downloaded model weights (OmniVoice or smart-turn)
```

- [ ] **Step 4: Write the failing test for the graph patch and loader**

Create `tests/test_turn_detection.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.gpu
def test_torch_model_matches_onnxruntime_reference():
    import onnx
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download

    from turn_detection import TURN_MODEL_FILE, TURN_MODEL_REPO, _load_torch_model

    onnx_path = hf_hub_download(repo_id=TURN_MODEL_REPO, filename=TURN_MODEL_FILE)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    torch_model = _load_torch_model(torch.device("cpu"))

    rng = np.random.default_rng(0)
    for _ in range(3):
        sample = rng.standard_normal((1, 80, 800)).astype(np.float32)

        onnx_out = session.run(None, {"input_features": sample})[0]
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(sample))
        if isinstance(torch_out, dict):
            torch_out = next(iter(torch_out.values()))
        torch_out = torch_out.numpy()

        assert np.allclose(onnx_out, torch_out, atol=1e-4)
```

- [ ] **Step 5: Run it to verify it fails**

Run: `pytest tests/test_turn_detection.py -v -m gpu`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_detection'`

- [ ] **Step 6: Implement `turn_detection.py`'s loader**

Create `turn_detection.py`:

```python
"""Torch-native turn detection using pipecat-ai/smart-turn v3.

Smart-turn v3's official checkpoint (pipecat-ai/smart-turn-v3 on the
Hugging Face Hub) ships as ONNX only -- there is no PyTorch/safetensors
checkpoint for this version. This module converts the published ONNX graph
to a torch.nn.Module with onnx2torch once at startup, so inference stays
torch-native and shares the GPU with OmniVoice instead of pulling in
onnxruntime as a second inference runtime.

Source: https://github.com/pipecat-ai/smart-turn (BSD-2-Clause)
"""

from __future__ import annotations

import onnx
import torch
from huggingface_hub import hf_hub_download
from onnx2torch import convert

TURN_MODEL_REPO = "pipecat-ai/smart-turn-v3"
TURN_MODEL_FILE = "smart-turn-v3.2-gpu.onnx"


def _patch_reshape_allowzero(model: onnx.ModelProto) -> onnx.ModelProto:
    """Flip allowzero=1 to 0 on this graph's Reshape nodes.

    onnx2torch 1.5.15 raises NotImplementedError on allowzero=1. Clearing
    it is safe for this specific graph: none of its Reshape target shapes
    are ever literally 0 (they're fixed-size attention-head reshapes for
    this model's fixed 8s/16kHz input), so allowzero=0 and allowzero=1
    behave identically for every shape this graph actually produces.
    Verified against pipecat-ai/smart-turn-v3.2-gpu.onnx by comparing the
    converted torch model's output to ONNX Runtime's (see
    tests/test_turn_detection.py).
    """
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in node.attribute:
            if attr.name == "allowzero" and attr.i == 1:
                attr.i = 0
    return model


def _load_torch_model(device: torch.device) -> torch.nn.Module:
    onnx_path = hf_hub_download(repo_id=TURN_MODEL_REPO, filename=TURN_MODEL_FILE)
    onnx_model = onnx.load(onnx_path)
    onnx_model = _patch_reshape_allowzero(onnx_model)

    torch_model = convert(onnx_model)
    torch_model.eval()
    return torch_model.to(device)
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_turn_detection.py -v -m gpu`
Expected: PASS (this downloads ~32MB on first run; subsequent runs use the Hugging Face cache)

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pytest.ini turn_detection.py tests/test_turn_detection.py
git commit -m "feat: torch-native loader for smart-turn-v3 via onnx2torch"
```

---

### Task 2: Truncate-to-last-N-seconds helper

**Files:**
- Modify: `turn_detection.py`
- Modify: `tests/test_turn_detection.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 3): `turn_detection.SAMPLE_RATE: int`, `turn_detection.CHUNK_SECONDS: int`, `turn_detection._truncate_audio_to_last_n_seconds(audio: np.ndarray, n_seconds: int = CHUNK_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_turn_detection.py`:

```python
from turn_detection import _truncate_audio_to_last_n_seconds


def test_shorter_than_target_is_zero_padded_at_the_start():
    audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    result = _truncate_audio_to_last_n_seconds(audio, n_seconds=1, sample_rate=8)

    assert result.shape == (8,)
    assert np.array_equal(result[:4], np.zeros(4, dtype=np.float32))
    assert np.array_equal(result[4:], audio)


def test_longer_than_target_keeps_the_tail():
    audio = np.arange(10, dtype=np.float32)

    result = _truncate_audio_to_last_n_seconds(audio, n_seconds=1, sample_rate=6)

    assert np.array_equal(result, audio[-6:])


def test_exact_length_is_unchanged():
    audio = np.arange(6, dtype=np.float32)

    result = _truncate_audio_to_last_n_seconds(audio, n_seconds=1, sample_rate=6)

    assert np.array_equal(result, audio)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_turn_detection.py -v -k "shorter_than_target or longer_than_target or exact_length"`
Expected: FAIL with `ImportError: cannot import name '_truncate_audio_to_last_n_seconds'`

- [ ] **Step 3: Implement the helper**

Add to `turn_detection.py`, near the top constants (after `TURN_MODEL_FILE`):

```python
import numpy as np

CHUNK_SECONDS = 8
SAMPLE_RATE = 16000
```

And add the function anywhere after `_load_torch_model`:

```python
def _truncate_audio_to_last_n_seconds(
    audio: np.ndarray,
    n_seconds: int = CHUNK_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Keep the end of `audio`, zero-padded at the start if it's shorter.

    Vendored from smart-turn's audio_utils.py (BSD-2-Clause): the most
    recent audio is what matters for a turn-completion decision, so a long
    clip should be truncated from the front, not the back.
    """
    target_len = n_seconds * sample_rate
    if len(audio) >= target_len:
        return audio[-target_len:]

    padded = np.zeros(target_len, dtype=np.float32)
    padded[-len(audio):] = audio
    return padded
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_turn_detection.py -v -k "shorter_than_target or longer_than_target or exact_length"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add turn_detection.py tests/test_turn_detection.py
git commit -m "feat: truncate-keep-tail audio helper for turn detection"
```

---

### Task 3: `TurnDetectionService`

**Files:**
- Modify: `turn_detection.py`
- Modify: `tests/test_turn_detection.py`

**Interfaces:**
- Consumes: `turn_detection._load_torch_model` (Task 1), `turn_detection._truncate_audio_to_last_n_seconds`, `turn_detection.SAMPLE_RATE`, `turn_detection.CHUNK_SECONDS` (Task 2).
- Produces (used by Task 5): `turn_detection.TurnDetectionService` with `.warmup() -> None`, `.predict(audio: np.ndarray) -> dict[str, float | int]`, `.is_ready: bool` (property).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_turn_detection.py`:

```python
import torch

from turn_detection import TurnDetectionService


class _ConstantModel:
    """Stands in for the converted torch model: always returns a fixed probability."""

    def __init__(self, probability: float) -> None:
        self.calls: list[torch.Tensor] = []
        self._probability = probability

    def __call__(self, input_features: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_features)
        return torch.tensor([[self._probability]])


def _stubbed_service(probability: float) -> tuple[TurnDetectionService, _ConstantModel]:
    from transformers import WhisperFeatureExtractor

    service = TurnDetectionService()
    model = _ConstantModel(probability)
    service._model = model
    service._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    service._device = torch.device("cpu")
    return service, model


def test_predict_above_threshold_is_complete():
    service, model = _stubbed_service(probability=0.87)

    result = service.predict(np.zeros(16000, dtype=np.float32))

    assert result == {"prediction": 1, "probability": pytest.approx(0.87)}
    assert model.calls[0].shape == (1, 80, 800)  # 8s at 16kHz -> 800 mel frames


def test_predict_below_threshold_is_incomplete():
    service, _ = _stubbed_service(probability=0.2)

    result = service.predict(np.zeros(16000, dtype=np.float32))

    assert result["prediction"] == 0


def test_predict_before_warmup_raises():
    service = TurnDetectionService()

    with pytest.raises(RuntimeError):
        service.predict(np.zeros(16000, dtype=np.float32))


def test_is_ready_reflects_warmup_state():
    service = TurnDetectionService()
    assert service.is_ready is False

    service._model = object()
    assert service.is_ready is True


@pytest.mark.gpu
def test_predict_end_to_end_with_the_real_model():
    service = TurnDetectionService()
    service.warmup()

    result = service.predict(np.zeros(16000, dtype=np.float32))

    assert set(result) == {"prediction", "probability"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_turn_detection.py -v -k "predict or is_ready"`
Expected: FAIL with `ImportError: cannot import name 'TurnDetectionService'`

- [ ] **Step 3: Implement `TurnDetectionService`**

Add to `turn_detection.py`, add this import at the top alongside the others:

```python
from transformers import WhisperFeatureExtractor
```

Then append the class at the end of the file:

```python
class TurnDetectionService:
    """Loads smart-turn-v3 once and predicts turn completion from audio."""

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._feature_extractor: WhisperFeatureExtractor | None = None
        self._device: torch.device | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        if self.is_ready:
            return

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = _load_torch_model(device)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=CHUNK_SECONDS)
        self._device = device

    def predict(self, audio: np.ndarray) -> dict[str, float | int]:
        if self._model is None or self._feature_extractor is None or self._device is None:
            raise RuntimeError("TurnDetectionService.warmup() must run before predict()")

        audio = _truncate_audio_to_last_n_seconds(audio.astype(np.float32))
        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=CHUNK_SECONDS * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.to(self._device)

        with torch.no_grad():
            output = self._model(input_features)
        if isinstance(output, dict):
            output = next(iter(output.values()))
        elif isinstance(output, (list, tuple)):
            output = output[0]

        probability = float(output.reshape(-1)[0].item())
        return {"prediction": 1 if probability > 0.5 else 0, "probability": probability}
```

- [ ] **Step 4: Run the fast tests to verify they pass**

Run: `pytest tests/test_turn_detection.py -v -k "predict or is_ready"`
Expected: PASS for all but `test_predict_end_to_end_with_the_real_model` (skipped by default, `pytest.ini` excludes `gpu`-marked tests)

- [ ] **Step 5: Run the full file including the gpu-marked tests**

Run: `pytest tests/test_turn_detection.py -v -m gpu`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add turn_detection.py tests/test_turn_detection.py
git commit -m "feat: TurnDetectionService.predict()"
```

---

### Task 4: Audio-upload decoding helpers

**Files:**
- Modify: `streaming_api_omnivoice.py:1-24` (imports)
- Create: `tests/test_turn_endpoint.py`

**Interfaces:**
- Consumes: nothing from other tasks (only `audio_formats.wav_header`, already in the codebase).
- Produces (used by Task 5): `_downmix_to_mono(waveform: torch.Tensor) -> torch.Tensor`, `_resample_if_needed(waveform: torch.Tensor, sample_rate: int, target_rate: int) -> torch.Tensor`, `_decode_audio_upload(data: bytes) -> np.ndarray` (all module-level functions in `streaming_api_omnivoice.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_turn_endpoint.py`:

```python
from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from audio_formats import wav_header
from streaming_api_omnivoice import _decode_audio_upload, _downmix_to_mono, _resample_if_needed


def _mono_wav_bytes(samples: list[int], sample_rate: int) -> bytes:
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    return wav_header(sample_rate, data_size=len(pcm)) + pcm


def test_downmix_to_mono_is_a_noop_for_mono_input():
    waveform = torch.tensor([[0.1, 0.2, 0.3]])
    assert torch.equal(_downmix_to_mono(waveform), waveform[0])


def test_downmix_to_mono_averages_channels():
    waveform = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    assert torch.allclose(_downmix_to_mono(waveform), torch.zeros(2))


def test_resample_if_needed_is_a_noop_at_target_rate():
    waveform = torch.tensor([0.1, 0.2, 0.3])
    assert torch.equal(_resample_if_needed(waveform, 16000, 16000), waveform)


def test_resample_if_needed_changes_length_at_a_different_rate():
    waveform = torch.zeros(8000)  # 1s at 8kHz

    result = _resample_if_needed(waveform, 8000, 16000)

    assert result.shape[0] == 16000


def test_decode_audio_upload_reads_a_16khz_mono_wav():
    samples = [0, 1000, -1000, 500] * 100
    data = _mono_wav_bytes(samples, sample_rate=16000)

    audio = _decode_audio_upload(data)

    assert audio.dtype == np.float32
    assert audio.shape[0] == len(samples)


def test_decode_audio_upload_resamples_to_16khz():
    samples = [0, 1000, -1000, 500] * 100  # 400 samples at 8kHz
    data = _mono_wav_bytes(samples, sample_rate=8000)

    audio = _decode_audio_upload(data)

    assert audio.shape[0] == 800  # resampled to 16kHz


def test_decode_audio_upload_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        _decode_audio_upload(b"not audio data")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_turn_endpoint.py -v`
Expected: FAIL with `ImportError: cannot import name '_decode_audio_upload'`

- [ ] **Step 3: Implement the decode helpers**

In `streaming_api_omnivoice.py`, change the import block at the top (lines 1-24) by adding `io` to the stdlib imports and `torchaudio` next to the existing `torch` import:

```python
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from omnivoice import OmniVoice
from pydantic import BaseModel, Field

from admission import AdmissionControl
from audio_formats import SUPPORTED_FORMATS, AudioEncoder, create_encoder
from batch_scheduler import BatchScheduler, GenerationJob
from turn_detection import SAMPLE_RATE, TurnDetectionService
from voices import VOICE_METADATA
from ws_session import SessionConfig, WebSocketSpeechSession
```

Then add these three functions right after the `MODEL_ID`/`STREAM_TEXT_CHUNK_SIZE`/warmup-constant block (after line 33, before `logger = logging.getLogger(__name__)`):

```python
def _downmix_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] == 1:
        return waveform[0]
    return waveform.mean(dim=0)


def _resample_if_needed(waveform: torch.Tensor, sample_rate: int, target_rate: int) -> torch.Tensor:
    if sample_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_rate)


def _decode_audio_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded audio file into mono float32 PCM at SAMPLE_RATE."""
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc

    waveform = _downmix_to_mono(waveform)
    waveform = _resample_if_needed(waveform, sample_rate, SAMPLE_RATE)
    return waveform.numpy().astype(np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_turn_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streaming_api_omnivoice.py tests/test_turn_endpoint.py
git commit -m "feat: decode multipart audio uploads to mono 16kHz PCM"
```

---

### Task 5: `POST /v1/turn/predict` endpoint, startup wiring, health field

**Files:**
- Modify: `streaming_api_omnivoice.py:388` (service instantiation), `:391-393` (startup event), `:403-409` (health endpoint), end of file (new endpoint)
- Modify: `tests/conftest.py:57-81` (`client` fixture)
- Modify: `tests/test_turn_endpoint.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `turn_detection.TurnDetectionService` (Task 3), `_decode_audio_upload` (Task 4).
- Produces: nothing further downstream — this is the last task.

- [ ] **Step 1: Write the failing tests**

First, extend the shared `client` fixture in `tests/conftest.py` so it stubs the turn model the same way it already stubs OmniVoice — otherwise every test using `client` would try to download the real smart-turn weights on `TestClient` startup. Change:

```python
    fake = FakeOmniVoice()
    api.service.scheduler = batch_scheduler.BatchScheduler(model_factory=lambda: fake)
    api.service.scheduler.start()
    assert api.service.scheduler.wait_ready(timeout=5)
    api.service.admission = AdmissionControl(max_streams=1)

    with TestClient(api.app) as test_client:
```

to:

```python
    fake = FakeOmniVoice()
    api.service.scheduler = batch_scheduler.BatchScheduler(model_factory=lambda: fake)
    api.service.scheduler.start()
    assert api.service.scheduler.wait_ready(timeout=5)
    api.service.admission = AdmissionControl(max_streams=1)

    api.turn_service._model = object()
    api.turn_service.predict = lambda audio: {"prediction": 1, "probability": 0.91}

    with TestClient(api.app) as test_client:
```

Then append to `tests/test_turn_endpoint.py`:

```python
def test_predict_turn_returns_prediction_and_probability(client):
    response = client.post(
        "/v1/turn/predict",
        files={"file": ("clip.wav", _mono_wav_bytes([0] * 1600, sample_rate=16000), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "probability": 0.91}


def test_predict_turn_rejects_empty_file(client):
    response = client.post(
        "/v1/turn/predict",
        files={"file": ("clip.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400


def test_predict_turn_rejects_undecodable_audio(client):
    response = client.post(
        "/v1/turn/predict",
        files={"file": ("clip.wav", b"not a real wav file", "audio/wav")},
    )

    assert response.status_code == 400


def test_predict_turn_rejects_oversized_upload(client, monkeypatch):
    import streaming_api_omnivoice as api

    monkeypatch.setattr(api, "MAX_TURN_UPLOAD_BYTES", 10)

    response = client.post(
        "/v1/turn/predict",
        files={"file": ("clip.wav", b"x" * 11, "audio/wav")},
    )

    assert response.status_code == 400


def test_health_reports_turn_model_loaded(client):
    response = client.get("/health")

    assert response.json()["turn_model_loaded"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_turn_endpoint.py -v`
Expected: FAIL — `/v1/turn/predict` doesn't exist yet (404s), and `AttributeError: module 'streaming_api_omnivoice' has no attribute 'turn_service'` from the fixture change.

- [ ] **Step 3: Wire up the service, startup, health, and endpoint**

In `streaming_api_omnivoice.py`, change line 388 from:

```python
service = OmniVoiceStreamingService()
```

to:

```python
service = OmniVoiceStreamingService()
turn_service = TurnDetectionService()
MAX_TURN_UPLOAD_BYTES = 25 * 1024 * 1024
```

Change the startup handler (lines 391-393) from:

```python
@app.on_event("startup")
def warmup_model() -> None:
    service.warmup()
```

to:

```python
@app.on_event("startup")
def warmup_model() -> None:
    service.warmup()
    turn_service.warmup()
```

Change the health endpoint (lines 403-409) from:

```python
@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": service.scheduler.sampling_rate > 0,
        "cuda_available": torch.cuda.is_available(),
    }
```

to:

```python
@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": service.scheduler.sampling_rate > 0,
        "turn_model_loaded": turn_service.is_ready,
        "cuda_available": torch.cuda.is_available(),
    }
```

Then add the endpoint at the end of the file:

```python
@app.post("/v1/turn/predict")
def predict_turn(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="file must not be empty")
    if len(data) > MAX_TURN_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"file exceeds the {MAX_TURN_UPLOAD_BYTES} byte limit",
        )

    try:
        audio = _decode_audio_upload(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return turn_service.predict(audio)
```

Finally, add `File` and `UploadFile` to the fastapi import at the top of the file — change:

```python
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket
```

to:

```python
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, WebSocket
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_turn_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Run the full fast test suite to check for regressions**

Run: `pytest -v`
Expected: PASS (gpu-marked tests are skipped by `pytest.ini`'s default `-m "not gpu"`)

- [ ] **Step 6: Document the endpoint in README.md**

Add a new section to `README.md`, after the existing "## WebSocket streaming" section (following the same style as the other endpoint docs — purpose, curl example, request/response shape):

```markdown
## Turn detection

`POST /v1/turn/predict` answers one question: has the speaker finished
their turn? It runs [smart-turn-v3](https://github.com/pipecat-ai/smart-turn)
(Whisper-tiny encoder + classifier head) as a torch-native model — the
official checkpoint is ONNX-only, so it's converted to a `torch.nn.Module`
once at startup and shares the GPU with OmniVoice.

```sh
curl -X POST http://localhost:9000/v1/turn/predict \
  --form 'file=@clip.mp3'
```

Send an audio file (any format `torchaudio`'s ffmpeg backend can decode —
mp3, wav, m4a, ...) as multipart form-data in a `file` field, same
convention as OpenAI-style transcription endpoints. Up to the last 8
seconds of audio is used; shorter clips are zero-padded at the start,
longer ones are truncated to keep the most recent audio.

Response:

```json
{"prediction": 1, "probability": 0.9231}
```

`prediction` is `1` when the turn is complete, `0` when it's not (i.e.
`probability > 0.5`). A `400` is returned for an empty file, an oversized
upload (>25MB), or audio that can't be decoded.
```

- [ ] **Step 7: Commit**

```bash
git add streaming_api_omnivoice.py tests/conftest.py tests/test_turn_endpoint.py README.md
git commit -m "feat: add POST /v1/turn/predict"
```

---

## Self-Review Notes

- **Spec coverage:** model source/onnx2torch conversion (Task 1), truncate-keep-tail preprocessing (Task 2), `TurnDetectionService` (Task 3), multipart decode via torchaudio (Task 4), endpoint + startup + health + size cap + error handling + README (Task 5), test coverage matching existing `tests/` conventions (all tasks) — every section of the design doc has a corresponding task.
- **Type consistency:** `TurnDetectionService.predict` returns `dict[str, float | int]` consistently across Task 3's definition and Task 5's endpoint return type; `_decode_audio_upload` returns `np.ndarray` consistently between Task 4's definition and Task 5's usage; `SAMPLE_RATE`/`CHUNK_SECONDS` are defined once in Task 2 and only ever imported, never redefined.
- **No placeholders:** every step has literal code, not descriptions of code.
