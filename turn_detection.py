"""Turn detection using pipecat-ai/smart-turn v3 (ONNX Runtime).

Smart-turn v3's official checkpoint (pipecat-ai/smart-turn-v3 on the
Hugging Face Hub) ships as ONNX only. Inference runs through onnxruntime's
CUDAExecutionProvider -- the same runtime smart-turn's own inference.py
uses -- sharing the GPU alongside torch/OmniVoice.

Preprocessing (truncate-keep-the-tail, WhisperFeatureExtractor) mirrors
smart-turn's own audio_utils.py / inference.py:
https://github.com/pipecat-ai/smart-turn (BSD-2-Clause)
"""

from __future__ import annotations

import io

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

TURN_MODEL_REPO = "pipecat-ai/smart-turn-v3"
TURN_MODEL_FILE = "smart-turn-v3.2-gpu.onnx"
CHUNK_SECONDS = 8
SAMPLE_RATE = 16000


def _load_onnx_session() -> ort.InferenceSession:
    onnx_path = hf_hub_download(repo_id=TURN_MODEL_REPO, filename=TURN_MODEL_FILE)

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(providers)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)


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


def _downmix_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] == 1:
        return waveform[0]
    return waveform.mean(dim=0)


def _resample_if_needed(waveform: torch.Tensor, sample_rate: int, target_rate: int) -> torch.Tensor:
    if sample_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_rate)


def decode_audio_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded audio file into mono float32 PCM at SAMPLE_RATE.

    Accepts any container torchaudio's ffmpeg backend can demux (mp3, wav,
    m4a, ...). Raises ValueError for empty/undecodable input.
    """
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc

    waveform = _downmix_to_mono(waveform)
    waveform = _resample_if_needed(waveform, sample_rate, SAMPLE_RATE)
    return waveform.numpy().astype(np.float32)


class TurnDetectionService:
    """Loads smart-turn-v3 once and predicts turn completion from audio."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._feature_extractor: WhisperFeatureExtractor | None = None

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def warmup(self) -> None:
        if self.is_ready:
            return

        self._session = _load_onnx_session()
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=CHUNK_SECONDS)

        # onnxruntime's CUDAExecutionProvider compiles/selects kernels lazily
        # on the first real session.run(), not at session construction. Pay
        # that cost here (~700ms) instead of on a user's first request.
        self.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def predict(self, audio: np.ndarray) -> dict[str, float | int]:
        if self._session is None or self._feature_extractor is None:
            raise RuntimeError("TurnDetectionService.warmup() must run before predict()")

        audio = _truncate_audio_to_last_n_seconds(audio.astype(np.float32))
        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=CHUNK_SECONDS * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.astype(np.float32)

        outputs = self._session.run(None, {"input_features": input_features})
        probability = float(outputs[0].reshape(-1)[0])

        return {"prediction": 1 if probability > 0.5 else 0, "probability": probability}


turn_service = TurnDetectionService()
