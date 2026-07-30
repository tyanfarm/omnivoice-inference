from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from omnivoice import OmniVoice
from pydantic import BaseModel, Field

from admission import AdmissionControl
from audio_formats import SUPPORTED_FORMATS, AudioEncoder, create_encoder
from batch_scheduler import BatchScheduler, GenerationJob
from voices import VOICE_METADATA
from ws_session import SessionConfig, WebSocketSpeechSession

BASE_DIR = Path(__file__).resolve().parent
PLAYER_PATH = BASE_DIR / "streaming_player.html"
VOICES_DIR = BASE_DIR / "voices"
MODEL_ID = "k2-fsa/OmniVoice"
STREAM_TEXT_CHUNK_SIZE = 240
WARMUP_TEXT = "Warm up."
WARMUP_REF_AUDIO = "voices/af_heart.mp3"
WARMUP_REF_TEXT = (
    "Human just went farther from earth than ever before. "
    "This was the mission to go back to the moon, with the goal of "
    "eventually establishing a moon colony."
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OmniVoice Streaming Test API",
    # No published schema: /docs, /redoc, and /openapi.json all 404.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

class StreamRequest(BaseModel):
    text: str
    voice_id: str = "af_heart"
    language: str | None = None
    instruct: str | None = None
    speed: float | None = 1.0
    num_step: int = Field(default=16, ge=1)
    denoise: bool = False
    postprocess_output: bool = False
    chunk_chars: int = Field(default=120, ge=40, le=600)


class SpeechRequest(BaseModel):
    """Body of POST /v1/audio/speech: OpenAI's `input`, `voice`, `response_format`.

    `input` is unbounded — text_to_speech_stream already splits it into chunks,
    so length costs time rather than memory. Everything else — speed, chunking,
    diffusion steps — comes from StreamRequest's defaults. Unrecognised fields
    an OpenAI client sends (`model`, `instructions`, ...) are ignored rather
    than rejected, so a stock client still gets audio.

    `response_format` is a plain str, not a Literal, so an unsupported value
    comes back as an OpenAI-shaped 400 naming the formats that do work instead
    of a bare pydantic 422.
    """

    input: str = Field(..., min_length=1)
    voice: str = "vf_phuong"
    response_format: str = "mp3"

    def to_stream_request(self) -> StreamRequest:
        return StreamRequest(text=self.input, voice_id=self.voice)


class OmniVoiceStreamingService:
    def __init__(self) -> None:
        self.scheduler = BatchScheduler(model_factory=self._load_model)
        self.admission = AdmissionControl()

    @staticmethod
    def _load_model() -> OmniVoice:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        return OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)

    def warmup(self) -> None:
        if self.scheduler.sampling_rate > 0:
            return

        self.scheduler.start()
        if not self.scheduler.wait_ready(timeout=600):
            raise RuntimeError("OmniVoice model failed to load within 600s")

        try:
            ref_audio_path = self._resolve_ref_audio_path(WARMUP_REF_AUDIO)
        except FileNotFoundError:
            logger.warning(
                "Warmup reference audio not found at %s; skipping warmup generation",
                WARMUP_REF_AUDIO,
            )
            return

        if ref_audio_path is None:
            return

        job = GenerationJob(
            text=WARMUP_TEXT,
            ref_audio=str(ref_audio_path),
            ref_text=WARMUP_REF_TEXT,
            language=None,
            speed=1.0,
            num_step=16,
            denoise=False,
            postprocess_output=False,
        )
        self.scheduler.submit(job)
        job.result(timeout=600)

    @staticmethod
    def _voice_to_payload(voice_path: Path) -> dict[str, str | None]:
        voice_id = voice_path.stem
        metadata = VOICE_METADATA.get(voice_id, {})
        relative_path = voice_path.relative_to(BASE_DIR).as_posix()
        return {
            "id": voice_id,
            "name": metadata.get("name", voice_id.replace("_", " ").title()),
            "native_language": metadata.get("native_language"),
            "gender": metadata.get("gender"),
            # "file_name": voice_path.name,
            # "ref_audio": relative_path,
            # "audio_url": f"/api/voices/{voice_id}/audio",
            # "ref_text": metadata.get("ref_text"),
        }

    def list_voices(self) -> list[dict[str, str | None]]:
        if not VOICES_DIR.exists():
            return []

        voice_files = sorted(
            path for path in VOICES_DIR.iterdir() if path.is_file() and path.suffix == ".mp3"
        )
        return [self._voice_to_payload(path) for path in voice_files]

    def list_voices_paginated(
        self,
        page: int,
        page_size: int,
    ) -> dict[str, object]:
        voices = self.list_voices()
        total = len(voices)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "count": len(voices[start:end]),
            "voices": voices[start:end],
        }

    def get_voice(self, voice_id: str) -> dict[str, str | None]:
        voice_path = VOICES_DIR / f"{voice_id}.mp3"
        if not voice_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
        return self._voice_to_payload(voice_path)

    def get_voice_audio_path(self, voice_id: str) -> Path:
        voice_path = VOICES_DIR / f"{voice_id}.mp3"
        if not voice_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
        return voice_path

    def get_voice_clone_config(self, voice_id: str) -> tuple[Path, str]:
        voice_path = self.get_voice_audio_path(voice_id)
        metadata = VOICE_METADATA.get(voice_id)
        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice metadata not found: {voice_id}",
            )

        ref_text = metadata.get("ref_text")
        if not ref_text:
            raise HTTPException(
                status_code=400,
                detail=f"Voice ref_text not configured: {voice_id}",
            )

        return voice_path, ref_text

    def split_text_for_streaming(
        self,
        text: str,
        max_chars: int | None = None,
    ) -> list[str]:
        max_chars = max_chars or STREAM_TEXT_CHUNK_SIZE
        sentence_parts = [
            part.strip()
            for part in re.findall(r'[^.!?]+[.!?]+(?:["\']+)?|[^.!?]+$', text)
            if part.strip()
        ]

        if not sentence_parts:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current_chunk = ""

        for part in sentence_parts:
            part_pieces = self._split_long_text_part(part, max_chars)

            for piece in part_pieces:
                if not current_chunk:
                    current_chunk = piece
                    continue

                candidate = f"{current_chunk} {piece}"
                if len(candidate) <= max_chars:
                    current_chunk = candidate
                    continue

                chunks.append(current_chunk)
                current_chunk = piece

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def _split_long_text_part(text_part: str, max_chars: int) -> list[str]:
        if len(text_part) <= max_chars:
            return [text_part]

        words = text_part.split()
        if not words:
            return []

        pieces: list[str] = []
        current_piece = words[0]

        for word in words[1:]:
            candidate = f"{current_piece} {word}"
            if len(candidate) <= max_chars:
                current_piece = candidate
                continue

            pieces.append(current_piece)
            current_piece = word

        if current_piece:
            pieces.append(current_piece)

        return pieces

    @staticmethod
    def _resolve_ref_audio_path(ref_audio: str | None) -> Path | None:
        if not ref_audio:
            return None

        ref_audio_path = (BASE_DIR / ref_audio).resolve()
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        return ref_audio_path

    @staticmethod
    def _audio_to_numpy(audio: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 2:
                audio = audio.mean(dim=0) if audio.size(0) > 1 else audio.squeeze(0)
            return audio.detach().cpu().numpy()

        chunk_np = np.asarray(audio)
        if chunk_np.ndim == 2:
            chunk_np = (
                chunk_np.mean(axis=0)
                if chunk_np.shape[0] > 1
                else np.squeeze(chunk_np, axis=0)
            )
        return chunk_np

    @staticmethod
    def audio_chunk_is_empty(audio: torch.Tensor | np.ndarray | None) -> bool:
        if audio is None:
            return True
        if isinstance(audio, torch.Tensor):
            return audio.numel() == 0
        return np.asarray(audio).size == 0

    @classmethod
    def audio_to_int16_bytes(cls, audio: torch.Tensor | np.ndarray) -> bytes:
        chunk_np = cls._audio_to_numpy(audio)
        audio_int16 = (np.clip(chunk_np, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    def chunk_speed(text: str, requested: float | None) -> float:
        """Playback speed for one chunk.

        Very short chunks are read at full speed; slowing a three-word phrase
        makes it sound broken. Public so the WebSocket session speaks text at
        the same rate the HTTP path does.
        """
        speed = requested if requested is not None else 0.8
        if len(text.split()) <= 4:
            return 1.0
        return speed

    def text_to_speech_stream(
        self,
        request: StreamRequest,
        slot=None,
        encoder: AudioEncoder | None = None,
    ) -> Generator[bytes, None, None]:
        """Stream encoded audio bytes generated from OmniVoice text chunks.

        `encoder` decides the wire format (mp3, wav, pcm) and defaults to mp3.
        """
        if encoder is None:
            encoder = create_encoder("mp3", self.scheduler.sampling_rate)

        pending: GenerationJob | None = None
        try:
            text = request.text.strip()
            if not text:
                return

            ref_audio_path, ref_text = self.get_voice_clone_config(request.voice_id)

            speed = self.chunk_speed(text, request.speed)

            leading = encoder.begin()
            if leading:
                yield leading

            for text_chunk in self.split_text_for_streaming(
                text,
                max_chars=request.chunk_chars,
            ):
                pending = GenerationJob(
                    text=text_chunk,
                    ref_audio=str(ref_audio_path),
                    ref_text=ref_text,
                    language=request.language,
                    speed=speed,
                    num_step=request.num_step,
                    denoise=request.denoise,
                    postprocess_output=request.postprocess_output,
                )
                self.scheduler.submit(pending)
                audio_chunk = pending.result()
                pending = None

                if self.audio_chunk_is_empty(audio_chunk):
                    continue

                encoded = encoder.encode(self.audio_to_int16_bytes(audio_chunk))
                if encoded:
                    yield encoded

            final_chunk = encoder.flush()
            if final_chunk:
                yield final_chunk
        finally:
            # Reached on normal completion and on GeneratorExit when the
            # client disconnects mid-stream.
            if pending is not None:
                pending.cancel()
            if slot is not None:
                slot.release()


service = OmniVoiceStreamingService()


@app.on_event("startup")
def warmup_model() -> None:
    service.warmup()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not PLAYER_PATH.exists():
        raise HTTPException(status_code=404, detail="streaming_player.html not found")
    return HTMLResponse(PLAYER_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": service.scheduler.sampling_rate > 0,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/voices")
def list_voices(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> dict[str, object]:
    return service.list_voices_paginated(page=page, page_size=page_size)


@app.get("/api/voices/{voice_id}")
def get_voice(voice_id: str) -> dict[str, str | None]:
    return service.get_voice(voice_id)


@app.get("/api/voices/{voice_id}/audio")
def get_voice_audio(voice_id: str) -> FileResponse:
    voice_path = service.get_voice_audio_path(voice_id)
    return FileResponse(voice_path, media_type="audio/mpeg", filename=voice_path.name)


def _start_audio_stream(
    request: StreamRequest,
    filename_stem: str,
    response_format: str = "mp3",
) -> StreamingResponse:
    """Admit one stream and hand back its response in `response_format`.

    Raises HTTPException — 400 for an unknown format, 500 when its encoder is
    not installed, 503 when the server is full, or whatever
    get_voice_clone_config raises — for the caller to shape into its own error
    format.

    The encoder is built before the slot is taken so a bad format cannot leak
    one, and before streaming starts so a missing backend is a clean error
    rather than a truncated body.
    """
    try:
        encoder = create_encoder(response_format, service.scheduler.sampling_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    slot = service.admission.try_acquire()
    if slot is None:
        raise HTTPException(
            status_code=503,
            detail="Too many concurrent streams; retry shortly",
            headers={"Retry-After": "5"},
        )

    try:
        # Resolve the voice before streaming starts so a bad voice_id is a
        # clean 404 rather than a truncated audio stream.
        service.get_voice_clone_config(request.voice_id)
    except HTTPException:
        slot.release()
        raise

    filename = f"{filename_stem}.{encoder.extension}"
    return StreamingResponse(
        service.text_to_speech_stream(request, slot=slot, encoder=encoder),
        media_type=encoder.media_type,
        headers={
            "Cache-Control": "no-cache",
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )


@app.post("/api/stream-mp3")
def stream_mp3_audio(request: StreamRequest) -> StreamingResponse:
    return _start_audio_stream(request, "omnivoice-stream")


def _openai_error(
    status_code: int,
    message: str,
    param: str | None = None,
    error_type: str = "invalid_request_error",
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """OpenAI's error envelope, which its client libraries parse for `message`."""
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": None,
            }
        },
    )


@app.post("/v1/audio/speech")
def create_speech(request: SpeechRequest) -> Response:
    """OpenAI-compatible text-to-speech.

    Same engine and admission control as /api/stream-mp3; only the request
    schema, the output format and the error envelope differ. Audio is streamed
    as it is generated rather than buffered, which OpenAI's client handles
    transparently.
    """
    if request.response_format not in SUPPORTED_FORMATS:
        # Checked here rather than in _start_audio_stream so the error names
        # response_format as the offending parameter instead of voice.
        return _openai_error(
            400,
            f"Unsupported response_format '{request.response_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            param="response_format",
        )

    try:
        return _start_audio_stream(
            request.to_stream_request(),
            "speech",
            response_format=request.response_format,
        )
    except HTTPException as exc:
        if exc.status_code == 503:
            return _openai_error(
                503,
                str(exc.detail),
                error_type="server_error",
                headers={"Retry-After": "5"},
            )
        # A missing voice is a 404 internally, but OpenAI reports an unknown
        # voice as a 400 invalid_request_error, so clients see what they expect.
        if exc.status_code in (400, 404):
            return _openai_error(400, str(exc.detail), param="voice")
        return _openai_error(exc.status_code, str(exc.detail), error_type="server_error")


async def _ws_reject(
    socket: WebSocket,
    code: int,
    message: str,
    param: str | None = None,
    error_type: str = "invalid_request_error",
) -> None:
    """Report a handshake failure in the OpenAI envelope, then close.

    The socket is already accepted, so there is no status code left to send —
    a structured frame plus a close code is the only way the client learns why.
    """
    await socket.send_json(
        {"error": {"message": message, "type": error_type, "param": param, "code": None}}
    )
    await socket.close(code=code)


@app.websocket("/v1/audio/speech/ws")
async def speech_websocket(
    socket: WebSocket,
    voice: str = Query(default="vf_phuong", min_length=1),
    response_format: str = Query(default="pcm"),
) -> None:
    """Stream text in, stream audio out, on one connection.

    Not an OpenAI endpoint — OpenAI has no WebSocket TTS — so this protocol is
    ours: JSON text frames in ({"type": "text"|"flush"|"done"}), binary audio
    frames out, one {"type": "done"} at the end. The session closes itself once
    the client has sent no text for OMNIVOICE_WS_IDLE_TIMEOUT_S.

    `voice` and `response_format` are the only knobs. Chunk size, speed, and
    diffusion steps come from SessionConfig's defaults, so a socket generates
    with the same settings an HTTP request does.

    A socket costs one AdmissionControl slot for its lifetime, the same pool
    HTTP streams draw from, so the two together stay under MAX_STREAMS.
    """
    await socket.accept()

    if response_format not in SUPPORTED_FORMATS:
        await _ws_reject(
            socket,
            1008,
            f"Unsupported response_format '{response_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            param="response_format",
        )
        return

    try:
        encoder = create_encoder(response_format, service.scheduler.sampling_rate)
    except RuntimeError as exc:
        await _ws_reject(socket, 1011, str(exc), error_type="server_error")
        return

    try:
        ref_audio_path, ref_text = service.get_voice_clone_config(voice)
    except HTTPException as exc:
        await _ws_reject(socket, 1008, str(exc.detail), param="voice")
        return

    # Taken last, so a rejected handshake can never leak one.
    slot = service.admission.try_acquire()
    if slot is None:
        await _ws_reject(
            socket,
            1013,
            "Too many concurrent streams; retry shortly",
            error_type="server_error",
        )
        return

    session = WebSocketSpeechSession(
        socket=socket,
        engine=service,
        config=SessionConfig(voice=voice, response_format=response_format),
        encoder=encoder,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )
    try:
        await session.run()
    finally:
        slot.release()


@app.get("/api/stream-mp3")
def stream_mp3_audio_get(
    text: str = Query(..., min_length=1),
    voice_id: str = Query(default="af_heart", min_length=1),
    language: str | None = None,
    instruct: str | None = None,
    speed: float | None = 0.9,
    num_step: int = Query(default=16, ge=1),
    denoise: bool = False,
    postprocess_output: bool = False,
    chunk_chars: int = Query(default=240, ge=40, le=600),
) -> StreamingResponse:
    request = StreamRequest(
        text=text,
        voice_id=voice_id,
        language=language,
        instruct=instruct,
        speed=speed,
        num_step=num_step,
        denoise=denoise,
        postprocess_output=postprocess_output,
        chunk_chars=chunk_chars,
    )
    return stream_mp3_audio(request)
