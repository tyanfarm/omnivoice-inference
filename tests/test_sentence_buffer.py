"""SentenceBuffer — deciding which streamed text is ready to speak.

Chunking itself belongs to the service's splitter; this only decides where
complete text ends, so a half-finished sentence is never sent to the model.
"""

from __future__ import annotations

import pytest

from streaming_api_omnivoice import OmniVoiceStreamingService
from ws_session import SentenceBuffer

# An instance, because the splitter is an instance method. Constructing the
# service builds a scheduler but never starts it, so no model is loaded.
SPLIT = OmniVoiceStreamingService().split_text_for_streaming


@pytest.fixture
def buffer():
    return SentenceBuffer(split=SPLIT, max_chars=240)


def test_a_partial_sentence_is_held_back(buffer):
    # Speaking half a clause would ruin prosody and cannot be un-said.
    assert buffer.append("Hello there") == []
    assert buffer.pending_chars == len("Hello there")


def test_a_terminator_releases_the_sentence(buffer):
    assert buffer.append("Hello there.") == ["Hello there."]
    assert buffer.pending_chars == 0


def test_the_tail_after_the_last_terminator_stays_buffered(buffer):
    assert buffer.append("One. Two. Three") == ["One. Two."]
    assert buffer.pending_chars == len(" Three")


def test_text_arriving_letter_by_letter_still_forms_one_sentence(buffer):
    emitted = [chunk for ch in "Hi there." for chunk in buffer.append(ch)]
    assert emitted == ["Hi there."]


def test_drain_releases_text_that_has_no_terminator(buffer):
    buffer.append("No terminator here")
    assert buffer.drain() == ["No terminator here"]
    assert buffer.pending_chars == 0


def test_drain_on_an_empty_buffer_emits_nothing(buffer):
    assert buffer.drain() == []


def test_a_closing_quote_stays_with_its_sentence(buffer):
    # Cutting right after the period would start the next chunk with a stray
    # quote, which the model would try to pronounce.
    assert buffer.append('He said "go." Then left') == ['He said "go."']


def test_a_long_clause_with_no_terminator_is_forced_out(buffer):
    # An LLM emitting a 600-character clause with no period must not stall the
    # stream waiting for one.
    emitted = buffer.append("word " * 120)
    assert emitted, "expected a forced emit once the buffer got long"
    assert buffer.pending_chars == 0


def test_a_forced_emit_still_respects_max_chars(buffer):
    emitted = buffer.append("word " * 200)
    assert emitted
    assert all(len(chunk) <= 240 for chunk in emitted)


TEXT = (
    "The mission went farther than ever before. It aimed to return to the "
    "moon. The goal was a permanent colony there. Everyone watched."
)


def stream_word_by_word(text, max_chars=120):
    buffer = SentenceBuffer(split=SPLIT, max_chars=max_chars)
    emitted: list[str] = []
    for word in text.split(" "):
        emitted += buffer.append(word + " ")
    return emitted + buffer.drain()


def test_streaming_loses_no_text_and_reorders_nothing():
    # Whatever the chunk boundaries turn out to be, the words the model speaks
    # must be exactly the words the client sent, in order.
    assert " ".join(stream_word_by_word(TEXT)).split() == TEXT.split()


def test_every_streamed_chunk_is_one_the_splitter_would_not_cut_further():
    # This is the real parity property. Streamed chunks cannot be *identical*
    # to one-shot chunks — the one-shot splitter packs sentences up to
    # max_chars, and mid-stream there is no way to know another sentence is
    # coming without stalling until it does. What must hold is that every
    # chunk is a boundary the splitter itself would have chosen: never a
    # half-sentence, never over max_chars.
    for chunk in stream_word_by_word(TEXT):
        assert SPLIT(chunk, max_chars=120) == [chunk], chunk


def test_streamed_chunks_end_on_sentence_boundaries():
    # A chunk ending mid-clause is the failure this class exists to prevent;
    # the model would speak it with a falling, finished-sounding intonation.
    chunks = stream_word_by_word(TEXT)
    assert all(chunk.endswith(".") for chunk in chunks), chunks


def test_streaming_emits_the_first_sentence_without_waiting_for_the_rest():
    # Latency to first audio is the entire reason for the WebSocket path. The
    # one-shot splitter would have packed these four sentences into one chunk.
    streamed = stream_word_by_word(TEXT)
    assert len(streamed) > len(SPLIT(TEXT, max_chars=120))
    assert streamed[0] == "The mission went farther than ever before."
