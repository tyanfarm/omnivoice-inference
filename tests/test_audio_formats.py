"""Encoders behind /v1/audio/speech's response_format."""

from __future__ import annotations

import struct

import pytest

from audio_formats import (
    SUPPORTED_FORMATS,
    WAV_HEADER_SIZE,
    Mp3Encoder,
    PcmEncoder,
    WavEncoder,
    create_encoder,
    wav_header,
)

SAMPLE_RATE = 24000
# Two frames of int16 mono silence.
PCM = struct.pack("<4h", 0, 1000, -1000, 0)


def test_wav_header_is_the_canonical_44_bytes():
    header = wav_header(SAMPLE_RATE)
    assert len(header) == WAV_HEADER_SIZE
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert header[36:40] == b"data"


def test_wav_header_declares_16_bit_mono_at_the_given_rate():
    # A wrong rate or channel count here plays back at the wrong pitch.
    fmt_tag, channels, rate, byte_rate, block_align, bits = struct.unpack(
        "<HHIIHH", wav_header(SAMPLE_RATE)[20:36]
    )
    assert (fmt_tag, channels, bits) == (1, 1, 16)
    assert rate == SAMPLE_RATE
    assert block_align == 2
    assert byte_rate == SAMPLE_RATE * 2


def test_wav_riff_size_stays_consistent_with_the_data_size():
    riff_size = struct.unpack("<I", wav_header(SAMPLE_RATE, data_size=100)[4:8])[0]
    assert riff_size == 36 + 100


def test_streaming_wav_sizes_do_not_overflow_32_bits():
    # Length is unknown mid-stream, so both fields carry the maximum. They must
    # still fit in the 4 bytes struct.pack gives them.
    header = wav_header(SAMPLE_RATE)
    riff_size, data_size = (
        struct.unpack("<I", header[4:8])[0],
        struct.unpack("<I", header[40:44])[0],
    )
    assert riff_size == 0xFFFFFFFF
    assert data_size == 0xFFFFFFFF - 36


def test_pcm_encoder_passes_samples_through_untouched():
    encoder = PcmEncoder(SAMPLE_RATE)
    assert encoder.begin() == b""
    assert encoder.encode(PCM) == PCM
    assert encoder.flush() == b""


def test_wav_encoder_is_a_header_followed_by_the_same_pcm():
    encoder = WavEncoder(SAMPLE_RATE)
    body = encoder.begin() + encoder.encode(PCM) + encoder.flush()
    assert body == wav_header(SAMPLE_RATE) + PCM


def test_mp3_encoder_emits_a_frame_after_flush():
    encoder = Mp3Encoder(SAMPLE_RATE)
    body = encoder.begin() + encoder.encode(PCM * 500) + encoder.flush()
    # 0xFF 0xEx/0xFx is the MPEG frame sync; LAME may also prepend an ID3 tag.
    assert body.startswith(b"ID3") or body[0] == 0xFF, body[:4]


@pytest.mark.parametrize(
    "response_format,media_type,extension",
    [
        ("mp3", "audio/mpeg", "mp3"),
        ("wav", "audio/wav", "wav"),
        ("pcm", "audio/pcm", "pcm"),
    ],
)
def test_create_encoder_carries_the_wire_metadata(response_format, media_type, extension):
    encoder = create_encoder(response_format, SAMPLE_RATE)
    assert encoder.media_type == media_type
    assert encoder.extension == extension


def test_create_encoder_rejects_a_format_this_server_cannot_produce():
    with pytest.raises(ValueError) as excinfo:
        create_encoder("opus", SAMPLE_RATE)
    # The message is handed to the caller verbatim, so it has to say what works.
    for name in SUPPORTED_FORMATS:
        assert name in str(excinfo.value)
