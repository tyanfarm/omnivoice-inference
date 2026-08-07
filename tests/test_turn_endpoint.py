"""HTTP-level behavior of POST /v1/turn/predict.

The `client` fixture (tests/conftest.py) stubs turn_service's session and
predict(), so no GPU/network is needed. Decoding and model logic have their
own unit tests in tests/test_turn_detection.py.
"""

from __future__ import annotations

import struct

from audio_formats import wav_header


def _mono_wav_bytes(samples: list[int], sample_rate: int) -> bytes:
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    return wav_header(sample_rate, data_size=len(pcm)) + pcm


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
