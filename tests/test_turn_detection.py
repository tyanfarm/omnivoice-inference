from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from audio_formats import wav_header
from turn_detection import (
    CHUNK_SECONDS,
    SAMPLE_RATE,
    TurnDetectionService,
    _downmix_to_mono,
    _resample_if_needed,
    _truncate_audio_to_last_n_seconds,
    decode_audio_upload,
)


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

    audio = decode_audio_upload(data)

    assert audio.dtype == np.float32
    assert audio.shape[0] == len(samples)


def test_decode_audio_upload_resamples_to_16khz():
    samples = [0, 1000, -1000, 500] * 100  # 400 samples at 8kHz
    data = _mono_wav_bytes(samples, sample_rate=8000)

    audio = decode_audio_upload(data)

    assert audio.shape[0] == 800  # resampled to 16kHz


def test_decode_audio_upload_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        decode_audio_upload(b"not audio data")


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


class FakeSession:
    """Stands in for ort.InferenceSession: records inputs, returns a fixed probability."""

    def __init__(self, probability: float) -> None:
        self.calls: list[dict] = []
        self._probability = probability

    def run(self, output_names, inputs):
        self.calls.append(inputs)
        return [np.array([[self._probability]], dtype=np.float32)]


def _stubbed_service(probability: float) -> tuple[TurnDetectionService, FakeSession]:
    from transformers import WhisperFeatureExtractor

    service = TurnDetectionService()
    session = FakeSession(probability)
    service._session = session
    service._feature_extractor = WhisperFeatureExtractor(chunk_length=CHUNK_SECONDS)
    return service, session


def test_predict_above_threshold_is_complete():
    service, session = _stubbed_service(probability=0.87)

    result = service.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert result == {"prediction": 1, "probability": pytest.approx(0.87)}
    fed = session.calls[0]["input_features"]
    assert fed.shape == (1, 80, CHUNK_SECONDS * 100)  # 8s -> 800 mel frames


def test_predict_below_threshold_is_incomplete():
    service, _ = _stubbed_service(probability=0.2)

    result = service.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert result["prediction"] == 0


def test_predict_before_warmup_raises():
    service = TurnDetectionService()

    with pytest.raises(RuntimeError):
        service.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))


def test_is_ready_reflects_warmup_state():
    service = TurnDetectionService()
    assert service.is_ready is False

    service._session = object()
    assert service.is_ready is True


@pytest.mark.gpu
def test_predict_end_to_end_with_the_real_model():
    service = TurnDetectionService()
    service.warmup()

    result = service.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert set(result) == {"prediction", "probability"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
