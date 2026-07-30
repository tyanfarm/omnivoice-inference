"""Command-line client for WS /v1/audio/speech/ws.

Streams text the way an LLM would — a word at a time, with a gap between
words — so the session's incremental sentence cutting and idle timeout get
exercised rather than bypassed by one big message.

    python ws_client.py "Hello there. This is a test." --out out.wav

Writes a playable WAV whichever transport format is used: `pcm` frames get a
header added locally, `wav` already carries one, `mp3` is written as-is to
out.mp3. Prints one line per audio frame so first-frame latency is visible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

from audio_formats import WAV_HEADER_SIZE, wav_header

SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="text to speak")
    parser.add_argument("--file", type=Path, help="read the text from a file instead")
    parser.add_argument("--url", default="ws://127.0.0.1:9000")
    parser.add_argument("--voice", default="vf_phuong")
    parser.add_argument(
        "--format", default="pcm", choices=("pcm", "wav", "mp3"), dest="response_format"
    )
    parser.add_argument("--out", type=Path, default=Path("ws_out.wav"))
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="seconds between words, standing in for an LLM's token gaps",
    )
    parser.add_argument(
        "--no-done",
        action="store_true",
        help="never send {'type':'done'} — let the idle timeout close the session",
    )
    return parser.parse_args()


async def feed(socket, text: str, delay: float, send_done: bool) -> None:
    """Send the text a word at a time, then optionally finish the utterance.

    A rejected handshake closes the socket while this is still writing, which
    is normal rather than an error — the reader picks up the reason.
    """
    try:
        for word in text.split():
            await socket.send(json.dumps({"type": "text", "text": word + " "}))
            if delay:
                await asyncio.sleep(delay)
        if send_done:
            await socket.send(json.dumps({"type": "done"}))
    except websockets.exceptions.ConnectionClosed:
        pass


async def run(args: argparse.Namespace) -> int:
    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    if not text:
        print("nothing to speak: pass text or --file", file=sys.stderr)
        return 2

    url = (
        f"{args.url}/v1/audio/speech/ws"
        f"?voice={args.voice}&response_format={args.response_format}"
    )
    print(f"connecting to {url}")

    audio = b""
    final: dict | None = None
    started = time.monotonic()
    first_frame: float | None = None

    socket = await websockets.connect(url)
    writer = asyncio.create_task(feed(socket, text, args.delay, not args.no_done))
    try:
        async for frame in socket:
            if isinstance(frame, bytes):
                if first_frame is None:
                    first_frame = time.monotonic() - started
                    print(f"first audio frame after {first_frame:.2f}s")
                audio += frame
                print(f"  frame {len(frame)} bytes (total {len(audio)})")
                continue
            final = json.loads(frame)
            break
    except websockets.exceptions.ConnectionClosed as exc:
        print(f"server closed the connection: {exc}")
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        await socket.close()

    print(f"server said: {final}")
    print(f"close code: {socket.close_code}")

    if final is not None and "error" in final:
        print(f"session failed: {final['error']['message']}", file=sys.stderr)
        return 1

    return write_audio(args, audio)


def write_audio(args: argparse.Namespace, audio: bytes) -> int:
    if not audio:
        print("no audio received", file=sys.stderr)
        return 1

    out = args.out
    if args.response_format == "mp3":
        # Compressed, so byte count says nothing about duration; ffprobe it.
        out = out.with_suffix(".mp3")
        out.write_bytes(audio)
        print(f"wrote {out} ({len(audio)} bytes)")
        return 0
    if args.response_format == "wav":
        # The server's header carries 0xFFFFFFFF sizes because the length is
        # unknown mid-stream. Rewrite it now that the length is known, so
        # strict players and `soxi` are happy too.
        body = audio[WAV_HEADER_SIZE:]
        out.write_bytes(wav_header(SAMPLE_RATE, data_size=len(body)) + body)
    else:
        out.write_bytes(wav_header(SAMPLE_RATE, data_size=len(audio)) + audio)

    seconds = len(audio) / 2 / SAMPLE_RATE
    print(f"wrote {out} ({len(audio)} bytes ~ {seconds:.2f}s of audio)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
