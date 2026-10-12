"""The x-api-key header, checked against X_API_KEY.

The `client` fixture (tests/conftest.py) sends the right key by default, so
these tests override or drop the header to see what a caller without it gets.
"""

from __future__ import annotations

import pytest
from starlette.testclient import WebSocketDenialResponse

PROTECTED = [
    ("get", "/api/voices"),
    ("get", "/api/voices/af_heart"),
    ("get", "/api/voices/af_heart/audio"),
    ("get", "/api/stream-mp3?text=job-1."),
    ("post", "/api/stream-mp3"),
    ("post", "/v1/audio/speech"),
    ("post", "/v1/turn/predict"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_protected_endpoints_refuse_a_missing_or_wrong_key(client, method, path, key):
    headers = {"x-api-key": key} if key else {}
    if key is None:
        client.headers.pop("x-api-key")
    response = client.request(method, path, headers=headers)
    assert response.status_code == 401


def test_the_right_key_is_let_through(client):
    assert client.get("/api/voices").status_code == 200


def test_health_needs_no_key(client):
    client.headers.pop("x-api-key")
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_the_websocket_refuses_the_handshake_without_the_right_key(client, key):
    # Refused before the upgrade, so the client never gets an open socket.
    client.headers.pop("x-api-key")
    headers = {"x-api-key": key} if key else {}
    url = "/v1/audio/speech/ws?voice=af_heart"
    with pytest.raises(WebSocketDenialResponse) as denied:
        with client.websocket_connect(url, headers=headers):
            pass
    assert denied.value.status_code == 401
    assert denied.value.json()["error"]["type"] == "authentication_error"


def test_the_server_refuses_to_start_without_a_key(monkeypatch):
    from fastapi.testclient import TestClient

    import streaming_api_omnivoice as api

    monkeypatch.setattr(api, "API_KEY", "")
    with pytest.raises(RuntimeError, match="X_API_KEY"):
        with TestClient(api.app):
            pass
