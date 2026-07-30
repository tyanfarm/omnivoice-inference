"""What the app does and does not expose.

The `client` fixture (tests/conftest.py) swaps in FakeOmniVoice, so no GPU.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_schema_and_docs_endpoints_are_not_served(client, path):
    # Disabling only openapi_url would 404 /docs as a side effect; asserting
    # all three means a later FastAPI upgrade cannot quietly re-expose one.
    assert client.get(path).status_code == 404


def test_the_real_endpoints_still_work(client):
    # Guards against disabling the schema by breaking app construction.
    assert client.get("/health").status_code == 200
    assert client.get("/api/voices").status_code == 200
