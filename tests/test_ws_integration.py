"""Real model, real audio over a WebSocket. Run with: pytest -m gpu"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def live_client():
    from fastapi.testclient import TestClient

    import streaming_api_omnivoice as api

    with TestClient(api.app) as client:
        yield client


def collect(socket):
    audio = b""
    while True:
        message = socket.receive()
        if message.get("bytes") is not None:
            audio += message["bytes"]
            continue
        return audio, json.loads(message["text"])


def test_streamed_sentences_return_playable_audio(live_client):
    url = "/v1/audio/speech/ws?voice=af_heart&response_format=wav"
    with live_client.websocket_connect(url) as socket:
        for sentence in ("Hello there. ", "This is a streaming test. ", "Goodbye."):
            socket.send_json({"type": "text", "text": sentence})
        socket.send_json({"type": "done"})
        audio, final = collect(socket)

    assert final["type"] == "done"
    assert audio.startswith(b"RIFF")
    # 24 kHz, 16-bit mono: three sentences must exceed a second of audio.
    assert (len(audio) - 44) / 2 / 24000 > 1.0


def test_two_concurrent_sessions_both_get_audio(live_client):
    url = "/v1/audio/speech/ws?voice=af_heart&response_format=pcm"
    with live_client.websocket_connect(url) as a, live_client.websocket_connect(
        url
    ) as b:
        for socket in (a, b):
            socket.send_json({"type": "text", "text": "Concurrent session test."})
            socket.send_json({"type": "done"})
        for socket in (a, b):
            audio, final = collect(socket)
            assert audio and final["type"] == "done"
