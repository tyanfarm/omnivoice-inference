"""WS /v1/audio/speech/ws — the streaming session.

Uses the `client` fixture's FakeOmniVoice, so no GPU. FakeOmniVoice needs a
"job-<n>" token in each text, which doubles as a way to assert ordering.
"""

from __future__ import annotations

import json
import re
import threading
import time

from audio_formats import WAV_HEADER_SIZE, wav_header


def connect(client, **params):
    query = "&".join(f"{k}={v}" for k, v in {"voice": "af_heart", **params}.items())
    return client.websocket_connect(f"/v1/audio/speech/ws?{query}")


def audio_frames(socket):
    """Collect binary frames up to the terminating JSON frame."""
    frames = []
    while True:
        message = socket.receive()
        if message.get("bytes") is not None:
            frames.append(message["bytes"])
            continue
        return frames, json.loads(message["text"])


def first_json(socket):
    return json.loads(socket.receive()["text"])


def spoken_texts(client):
    return [text for call in client.fake_model.calls for text in call["text"]]


def test_a_finished_sentence_produces_audio(client):
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)
    assert frames and all(frames)
    assert final["type"] == "done"


def test_flush_releases_a_sentence_that_has_no_terminator(client):
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1 with no terminator yet"})
        socket.send_json({"type": "flush"})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)
    assert frames, "flush should have released the partial sentence"
    assert final["type"] == "done"


def test_sentences_are_spoken_in_the_order_they_arrived(client):
    # Three separate messages, each already terminated, so each is released as
    # its own chunk. One message holding all three would be packed into a
    # single chunk and prove nothing about ordering.
    client.fake_model.calls.clear()
    with connect(client) as socket:
        for i in range(3):
            socket.send_json({"type": "text", "text": f"job-{i} speaks a sentence."})
        socket.send_json({"type": "done"})
        audio_frames(socket)

    spoken = spoken_texts(client)
    ids = [int(re.search(r"job-(\d+)", text).group(1)) for text in spoken]
    assert ids == [0, 1, 2], f"audio would play out of order: {spoken}"


def test_a_session_never_has_two_chunks_generating_at_once(client):
    # Sequential submission is what keeps one session's backlog from crowding
    # the shared queue, so no generate() call may hold two of its chunks. The
    # text has to exceed chunk_chars (240) to be split at all, since the
    # endpoint exposes no chunk_chars knob.
    client.fake_model.calls.clear()
    long_text = " ".join(f"job-{i} " + "filler words here " * 4 + "." for i in range(6))
    assert len(long_text) > 240, "text must be long enough to split"

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": long_text})
        socket.send_json({"type": "done"})
        audio_frames(socket)

    assert len(client.fake_model.calls) > 1, "text should have split into chunks"
    for call in client.fake_model.calls:
        assert len(call["text"]) == 1, f"session chunks were batched together: {call}"


def test_pcm_is_the_default_format(client):
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, _ = audio_frames(socket)
    assert not frames[0].startswith(b"RIFF")


def test_wav_sends_its_header_as_the_first_frame(client):
    with connect(client, response_format="wav") as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        frames, _ = audio_frames(socket)
    assert frames[0][:WAV_HEADER_SIZE] == wav_header(client.fake_model.sampling_rate)


def test_the_voice_from_the_query_string_is_the_one_used(client):
    client.fake_model.prompt_calls.clear()
    with connect(client, voice="am_michael") as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        audio_frames(socket)
    refs = {call[0] for call in client.fake_model.prompt_calls}
    assert refs == {str(client.voices_dir / "am_michael.mp3")}, refs


def test_generation_uses_the_same_defaults_as_the_http_endpoint(client):
    from streaming_api_omnivoice import StreamRequest

    defaults = StreamRequest(text="job-1")
    client.fake_model.calls.clear()
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        audio_frames(socket)

    call = client.fake_model.calls[0]
    assert call["num_step"] == defaults.num_step
    assert call["denoise"] == defaults.denoise
    assert call["postprocess_output"] == defaults.postprocess_output


def test_an_unknown_message_type_is_reported_not_ignored(client):
    with connect(client) as socket:
        socket.send_json({"type": "sing", "text": "job-1."})
        payload = first_json(socket)
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "type"


def test_a_generation_failure_is_reported_as_a_server_error(client):
    # The worker raising must not leave the client waiting on a socket that
    # never says anything again.
    client.fake_model.fail_texts = {"job-9."}
    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-9."})
        payload = first_json(socket)
    assert payload["error"]["type"] == "server_error"


def test_an_unknown_voice_is_reported_before_any_audio(client):
    with connect(client, voice="does-not-exist") as socket:
        payload = first_json(socket)
    assert payload["error"]["param"] == "voice"
    assert payload["error"]["type"] == "invalid_request_error"


def test_an_unsupported_format_is_reported(client):
    with connect(client, response_format="opus") as socket:
        payload = first_json(socket)
    assert payload["error"]["param"] == "response_format"
    for name in ("mp3", "wav", "pcm"):
        assert name in payload["error"]["message"]


def test_over_capacity_is_refused_with_a_server_error(client):
    import streaming_api_omnivoice as api

    held = api.service.admission.try_acquire()
    assert held is not None
    try:
        with connect(client) as socket:
            payload = first_json(socket)
        assert payload["error"]["type"] == "server_error"
    finally:
        held.release()


def test_the_slot_is_returned_when_the_session_ends(client):
    import streaming_api_omnivoice as api

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.send_json({"type": "done"})
        audio_frames(socket)
    assert api.service.admission.active == 0


def test_a_refused_handshake_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    with connect(client, voice="nope") as socket:
        first_json(socket)
    assert api.service.admission.active == 0


def test_a_client_disconnecting_mid_session_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        socket.receive()  # first audio frame, then walk away
    assert api.service.admission.active == 0


def test_the_session_closes_itself_after_the_idle_timeout(client, monkeypatch):
    # The client stops sending text and never says "done" — the common case
    # when an LLM stream just ends. The server must not hold the slot.
    import streaming_api_omnivoice as api
    import ws_session

    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 0.3)

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-1."})
        frames, final = audio_frames(socket)

    assert frames, "buffered audio should still be delivered"
    assert final["type"] == "done"
    assert api.service.admission.active == 0


def test_idle_flush_speaks_text_that_never_got_a_terminator(client, monkeypatch):
    # A trailing fragment must not be silently dropped when the stream ends.
    import ws_session

    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 0.3)
    client.fake_model.calls.clear()

    with connect(client) as socket:
        socket.send_json({"type": "text", "text": "job-2 without a period"})
        audio_frames(socket)

    assert any("job-2" in text for text in spoken_texts(client)), spoken_texts(client)


def test_the_idle_timer_resets_on_every_text_message(client, monkeypatch):
    # A slow LLM sending a word every 0.1s must not be cut off at 0.3s.
    import ws_session

    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 0.3)

    with connect(client) as socket:
        for i in range(5):
            socket.send_json({"type": "text", "text": f"job-{i} "})
            time.sleep(0.1)
        socket.send_json({"type": "text", "text": "end."})
        socket.send_json({"type": "done"})
        frames, final = audio_frames(socket)

    assert final["type"] == "done"
    assert frames


def test_the_hard_ceiling_ends_a_session_that_never_stops(client, monkeypatch):
    # A client trickling text just under the idle timeout forever would hold a
    # slot indefinitely; max_session_s is the backstop. The trickle runs on its
    # own thread so the main thread can block waiting for the server to close.
    import streaming_api_omnivoice as api
    import ws_session

    monkeypatch.setattr(ws_session, "WS_MAX_SESSION_S", 0.5)
    monkeypatch.setattr(ws_session, "WS_IDLE_TIMEOUT_S", 5.0)

    with connect(client) as socket:
        stop = threading.Event()

        def trickle():
            while not stop.is_set():
                try:
                    socket.send_json({"type": "text", "text": "job-1 "})
                except Exception:
                    return  # the server closed on us, which is the point
                time.sleep(0.1)

        writer = threading.Thread(target=trickle, daemon=True)
        writer.start()
        try:
            started = time.monotonic()
            _, final = audio_frames(socket)
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            writer.join(timeout=2)

    assert final["type"] == "done"
    assert elapsed < 3.0, f"ceiling did not end the session; waited {elapsed:.1f}s"
    assert api.service.admission.active == 0
