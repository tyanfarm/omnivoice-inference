"""POST /v1/audio/speech — the OpenAI-compatible surface.

The `client` fixture (tests/conftest.py) swaps in FakeOmniVoice, so these run
without a GPU. FakeOmniVoice only understands input text shaped "job-<n>".
"""

from __future__ import annotations

import pytest


def speak(client, **overrides):
    body = {"input": "job-1", "voice_id": "af_heart"}
    body.update(overrides)
    return client.post("/v1/audio/speech", json=body)


def test_minimal_request_returns_mp3(client):
    response = speak(client)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0


def test_voice_id_defaults_when_omitted(client):
    response = client.post("/v1/audio/speech", json={"input": "job-1"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("voice_id", ["af_heart", "am_michael", "vf_phuong"])
def test_every_configured_voice_is_reachable(client, voice_id):
    assert speak(client, voice_id=voice_id).status_code == 200


def test_the_voice_reaches_the_reference_audio_it_names(client):
    # A wrong lookup here would hand the caller a different speaker than the
    # one they asked for, with no error to notice.
    client.fake_model.prompt_calls.clear()
    assert speak(client, voice_id="am_michael").status_code == 200
    refs = {call[0] for call in client.fake_model.prompt_calls}
    assert refs == {str(client.voices_dir / "am_michael.mp3")}, refs


def test_generation_uses_stream_request_defaults(client):
    # The endpoint deliberately exposes no tuning knobs, so a request must
    # arrive at the model with StreamRequest's defaults, not with None or 0.
    from streaming_api_omnivoice import StreamRequest

    defaults = StreamRequest(text="job-1")
    client.fake_model.calls.clear()
    assert speak(client).status_code == 200

    call = client.fake_model.calls[0]
    assert call["num_step"] == defaults.num_step
    assert call["denoise"] == defaults.denoise
    assert call["postprocess_output"] == defaults.postprocess_output
    assert call["language"] == [defaults.language]


def test_unknown_voice_is_a_400_in_the_openai_error_envelope(client):
    response = speak(client, voice_id="does-not-exist")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "voice"
    assert error["message"]


def test_empty_input_is_rejected(client):
    assert speak(client, input="").status_code == 422


def test_extra_openai_fields_are_ignored_not_rejected(client):
    # A stock OpenAI client always sends model/response_format/speed and may
    # send instructions. None of them exist here, but rejecting the request
    # would break every such client, so they are dropped instead.
    response = speak(
        client,
        model="gpt-4o-mini-tts",
        response_format="wav",
        speed=2.5,
        instructions="Speak like a pirate.",
    )
    assert response.status_code == 200
    # Output is mp3 regardless of what response_format asked for.
    assert response.headers["content-type"] == "audio/mpeg"


def test_overload_returns_503_in_the_openai_error_envelope(client):
    import streaming_api_omnivoice as api

    held = api.service.admission.try_acquire()
    assert held is not None
    try:
        response = speak(client)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert response.json()["error"]["type"] == "server_error"
    finally:
        held.release()


def test_slot_is_returned_after_a_stream_completes(client):
    import streaming_api_omnivoice as api

    response = speak(client)
    assert response.status_code == 200
    _ = response.content
    assert api.service.admission.active == 0


def test_refused_request_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    assert speak(client, voice_id="does-not-exist").status_code == 400
    assert api.service.admission.active == 0
