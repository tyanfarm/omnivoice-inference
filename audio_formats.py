"""Output encoders for the streaming TTS endpoints.

Each encoder turns the int16 mono PCM the model produces into one wire format.
They hold per-request state, so build one encoder per request and drive it as
begin() -> encode()* -> flush().
"""

from __future__ import annotations

import struct
from typing import Protocol

try:
    import lameenc
except ImportError:
    lameenc = None

CHANNELS = 1
BITS_PER_SAMPLE = 16
MP3_BIT_RATE = 128
MP3_QUALITY = 2

WAV_HEADER_SIZE = 44
# Bytes of header that follow the RIFF size field.
_RIFF_HEADER_TAIL = 36
# A streamed WAV cannot declare a length it does not know yet, so the size
# fields carry the largest value that fits and players read to end-of-stream.
_UNKNOWN_SIZE = 0xFFFFFFFF


def wav_header(sample_rate: int, data_size: int | None = None) -> bytes:
    """A 44-byte canonical RIFF/WAVE header for 16-bit mono PCM."""
    if data_size is None:
        data_size = _UNKNOWN_SIZE - _RIFF_HEADER_TAIL

    bytes_per_frame = CHANNELS * BITS_PER_SAMPLE // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        _RIFF_HEADER_TAIL + data_size,
        b"WAVE",
        b"fmt ",
        16,  # size of the fmt chunk
        1,  # format tag: uncompressed PCM
        CHANNELS,
        sample_rate,
        sample_rate * bytes_per_frame,  # byte rate
        bytes_per_frame,  # block align
        BITS_PER_SAMPLE,
        b"data",
        data_size,
    )


class AudioEncoder(Protocol):
    media_type: str
    extension: str

    def begin(self) -> bytes:
        """Leading bytes of the stream, before any audio."""

    def encode(self, pcm: bytes) -> bytes:
        """Encode one chunk of int16 PCM. May return b"" while buffering."""

    def flush(self) -> bytes:
        """Trailing bytes once the last chunk has been encoded."""


class PcmEncoder:
    """Raw 16-bit little-endian PCM — OpenAI's `pcm`, headerless."""

    media_type = "audio/pcm"
    extension = "pcm"

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def begin(self) -> bytes:
        return b""

    def encode(self, pcm: bytes) -> bytes:
        return pcm

    def flush(self) -> bytes:
        return b""


class WavEncoder(PcmEncoder):
    """The same PCM behind a RIFF header with streaming (unknown) sizes."""

    media_type = "audio/wav"
    extension = "wav"

    def begin(self) -> bytes:
        return wav_header(self.sample_rate)


class Mp3Encoder:
    """MP3 via LAME. Buffers internally, so encode() can return b""."""

    media_type = "audio/mpeg"
    extension = "mp3"

    def __init__(self, sample_rate: int) -> None:
        if lameenc is None:
            raise RuntimeError("Install lameenc in the venv to use MP3 output")

        self.sample_rate = sample_rate
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(MP3_BIT_RATE)
        self._encoder.set_in_sample_rate(sample_rate)
        self._encoder.set_channels(CHANNELS)
        self._encoder.set_quality(MP3_QUALITY)

    def begin(self) -> bytes:
        return b""

    def encode(self, pcm: bytes) -> bytes:
        return bytes(self._encoder.encode(pcm))

    def flush(self) -> bytes:
        return bytes(self._encoder.flush())


ENCODERS: dict[str, type] = {
    "mp3": Mp3Encoder,
    "wav": WavEncoder,
    "pcm": PcmEncoder,
}
SUPPORTED_FORMATS = tuple(ENCODERS)


def create_encoder(response_format: str, sample_rate: int) -> AudioEncoder:
    """Build the encoder for `response_format`.

    Raises ValueError for a format this server does not produce, and
    RuntimeError when the format is known but its backend is not installed.
    """
    factory = ENCODERS.get(response_format)
    if factory is None:
        raise ValueError(
            f"Unsupported response_format '{response_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    return factory(sample_rate)
