from __future__ import annotations

import concurrent.futures

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gpu

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "The river runs slowly behind the distant mountains."
)


@pytest.fixture(scope="module")
def client():
    import streaming_api_omnivoice as api

    with TestClient(api.app) as test_client:
        yield test_client


def fetch(client, voice_id):
    response = client.post(
        "/api/stream-mp3",
        json={"text": TEXT, "voice_id": voice_id, "chunk_chars": 120},
    )
    assert response.status_code == 200, response.text
    return response.content


def test_four_concurrent_streams_all_return_playable_audio(client):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        payloads = list(pool.map(lambda _: fetch(client, "af_heart"), range(4)))

    for payload in payloads:
        assert len(payload) > 1000, "stream returned suspiciously little audio"
        # MP3 frames start with a sync word (11 set bits) or an ID3 tag.
        assert payload[:3] == b"ID3" or payload[0] == 0xFF


def test_different_voices_batch_together_and_differ(client):
    import voices as voice_module

    ids = list(voice_module.VOICE_METADATA)[:2]
    if len(ids) < 2:
        pytest.skip("need at least two configured voices")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda v: fetch(client, v), ids))

    assert first != second, "distinct voices produced identical audio"
