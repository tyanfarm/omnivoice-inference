"""POST /v1/audio/speech — the OpenAI-compatible surface.

The `client` fixture (tests/conftest.py) swaps in FakeOmniVoice, so these run
without a GPU. FakeOmniVoice only understands input text shaped "job-<n>".
"""

from __future__ import annotations

import pytest

from audio_formats import WAV_HEADER_SIZE, wav_header


def speak(client, **overrides):
    body = {"input": "job-1", "voice": "af_heart"}
    body.update(overrides)
    return client.post("/v1/audio/speech", json=body)


def test_minimal_request_returns_mp3(client):
    response = speak(client)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0


def test_voice_defaults_when_omitted(client):
    response = client.post("/v1/audio/speech", json={"input": "job-1"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("voice", ["af_heart", "am_michael", "vf_phuong"])
def test_every_configured_voice_is_reachable(client, voice):
    assert speak(client, voice=voice).status_code == 200


def test_the_voice_reaches_the_reference_audio_it_names(client):
    # A wrong lookup here would hand the caller a different speaker than the
    # one they asked for, with no error to notice.
    client.fake_model.prompt_calls.clear()
    assert speak(client, voice="am_michael").status_code == 200
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


@pytest.mark.parametrize(
    "response_format,media_type",
    [("mp3", "audio/mpeg"), ("wav", "audio/wav"), ("pcm", "audio/pcm")],
)
def test_each_supported_format_streams_with_its_own_media_type(
    client, response_format, media_type
):
    response = speak(client, response_format=response_format)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == media_type
    assert len(response.content) > 0


def test_the_filename_extension_follows_the_format(client):
    response = speak(client, response_format="wav")
    assert 'filename="speech.wav"' in response.headers["content-disposition"]


def test_wav_output_starts_with_a_riff_header_for_the_model_sample_rate(client):
    response = speak(client, response_format="wav")
    assert response.status_code == 200, response.text
    header = response.content[:WAV_HEADER_SIZE]
    assert header == wav_header(client.fake_model.sampling_rate), header
    assert len(response.content) > WAV_HEADER_SIZE


def test_pcm_output_is_the_wav_payload_without_the_header(client):
    # Same text and voice, so the samples are identical; only the framing
    # differs. This is what makes `pcm` usable by a caller that adds its own.
    pcm = speak(client, response_format="pcm")
    wav = speak(client, response_format="wav")
    assert pcm.status_code == wav.status_code == 200
    assert not pcm.content.startswith(b"RIFF")
    assert pcm.content == wav.content[WAV_HEADER_SIZE:]
    assert len(pcm.content) % 2 == 0, "int16 samples must not be split"


def test_an_unsupported_format_is_a_400_naming_the_ones_that_work(client):
    response = speak(client, response_format="opus")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "response_format"
    for name in ("mp3", "wav", "pcm"):
        assert name in error["message"]


def test_an_unsupported_format_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    assert speak(client, response_format="opus").status_code == 400
    assert api.service.admission.active == 0


def test_unknown_voice_is_a_400_in_the_openai_error_envelope(client):
    response = speak(client, voice="does-not-exist")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "voice"
    assert error["message"]


def test_empty_input_is_rejected(client):
    assert speak(client, input="").status_code == 422


def test_extra_openai_fields_are_ignored_not_rejected(client):
    # A stock OpenAI client always sends model and speed and may send
    # instructions. None of them exist here, but rejecting the request would
    # break every such client, so they are dropped instead.
    response = speak(
        client,
        model="gpt-4o-mini-tts",
        speed=2.5,
        instructions="Speak like a pirate.",
    )
    assert response.status_code == 200
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

    assert speak(client, voice="does-not-exist").status_code == 400
    assert api.service.admission.active == 0
