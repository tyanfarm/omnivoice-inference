from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from omnivoice import OmniVoice
from pydantic import BaseModel, Field

try:
    import lameenc
except ImportError:
    lameenc = None

BASE_DIR = Path(__file__).resolve().parent
PLAYER_PATH = BASE_DIR / "streaming_player.html"
MODEL_ID = "k2-fsa/OmniVoice"
STREAM_TEXT_CHUNK_SIZE = 240
WARMUP_TEXT = "Warm up."
WARMUP_REF_AUDIO = "voices/am_michael.mp3"
WARMUP_REF_TEXT = (
    "Human just went farther from earth than ever before. "
    "This was the mission to go back to the moon, with the goal of "
    "eventually establishing a moon colony."
)

logger = logging.getLogger(__name__)

app = FastAPI(title="OmniVoice Streaming Test API")


class StreamRequest(BaseModel):
    text: str
    language: str | None = None
    ref_audio: str | None = "am_michael.mp3"
    ref_text: str | None = (
        "Human just went farther from earth than ever before. "
        "This was the mission to go back to the moon, with the goal of "
        "eventually establishing a moon colony."
    )
    instruct: str | None = None
    speed: float | None = 1.0
    num_step: int = Field(default=16, ge=1)
    denoise: bool = False
    postprocess_output: bool = False
    chunk_chars: int = Field(default=120, ge=40, le=600)


class OmniVoiceStreamingService:
    def __init__(self) -> None:
        self._model: OmniVoice | None = None
        self._lock = threading.Lock()
        self._voice_prompt_cache: dict[tuple[str, str], object] = {}

    def _get_model(self) -> OmniVoice:
        if self._model is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            self._model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
            )

        return self._model

    def warmup(self) -> None:
        with self._lock:
            model = self._get_model()
            generation_kwargs = {
                "text": WARMUP_TEXT,
                "speed": 1.0,
                "num_step": 16,
                "denoise": False,
                "postprocess_output": False,
            }

            try:
                ref_audio_path = self._resolve_ref_audio_path(WARMUP_REF_AUDIO)
                if ref_audio_path is not None:
                    generation_kwargs["voice_clone_prompt"] = (
                        self._get_voice_clone_prompt(
                            str(ref_audio_path),
                            WARMUP_REF_TEXT,
                        )
                    )
            except FileNotFoundError:
                logger.warning(
                    "Warmup reference audio not found at %s; continuing without voice clone prompt",
                    WARMUP_REF_AUDIO,
                )

            model.generate(**generation_kwargs)

    def _get_voice_clone_prompt(self, ref_audio: str, ref_text: str):
        key = (ref_audio, ref_text)
        if key not in self._voice_prompt_cache:
            model = self._get_model()
            self._voice_prompt_cache[key] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                preprocess_prompt=True,
            )
        return self._voice_prompt_cache[key]

    def _split_text_for_streaming(
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
    def _audio_tensor_to_int16_bytes(audio: torch.Tensor) -> bytes:
        if audio.dim() == 2:
            audio = audio.mean(dim=0) if audio.size(0) > 1 else audio.squeeze(0)

        chunk_np = audio.detach().cpu().numpy()
        audio_int16 = (np.clip(chunk_np, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def text_to_speech_stream(
        self,
        request: StreamRequest,
    ) -> Generator[bytes, None, None]:
        """Stream MP3 audio bytes generated from OmniVoice text chunks."""
        if lameenc is None:
            raise RuntimeError("lameenc is required for MP3 streaming")

        text = request.text.strip()
        if not text:
            return

        if request.ref_audio and not request.ref_text:
            raise ValueError("ref_text is required when using ref_audio")

        ref_audio_path = self._resolve_ref_audio_path(request.ref_audio)

        speed = request.speed if request.speed is not None else 0.8
        if len(text.split()) <= 4:
            speed = 1.0

        with self._lock:
            model = self._get_model()
            sample_rate = model.sampling_rate

            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(1)
            encoder.set_quality(2)

            voice_clone_prompt = None
            if ref_audio_path and request.ref_text:
                voice_clone_prompt = self._get_voice_clone_prompt(
                    str(ref_audio_path),
                    request.ref_text,
                )

            for text_chunk in self._split_text_for_streaming(
                text,
                max_chars=request.chunk_chars,
            ):
                generation_kwargs = {
                    "text": text_chunk,
                    "language": request.language,
                    "speed": speed,
                    "num_step": request.num_step,
                    "denoise": request.denoise,
                    "postprocess_output": request.postprocess_output,
                }

                if voice_clone_prompt is not None:
                    generation_kwargs["voice_clone_prompt"] = voice_clone_prompt
                elif request.instruct:
                    generation_kwargs["instruct"] = request.instruct

                audio_chunk = model.generate(**generation_kwargs)[0]
                if audio_chunk is None or audio_chunk.numel() == 0:
                    continue

                mp3_chunk = encoder.encode(
                    self._audio_tensor_to_int16_bytes(audio_chunk)
                )
                if mp3_chunk:
                    yield bytes(mp3_chunk)

            final_chunk = encoder.flush()
            if final_chunk:
                yield bytes(final_chunk)


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
        "model_loaded": service._model is not None,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/api/stream-mp3")
def stream_mp3_audio(request: StreamRequest) -> StreamingResponse:
    if lameenc is None:
        raise HTTPException(
            status_code=500,
            detail="Install lameenc in the venv to use MP3 streaming",
        )

    return StreamingResponse(
        service.text_to_speech_stream(request),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Content-Disposition": 'inline; filename="omnivoice-stream.mp3"',
        },
    )


@app.get("/api/stream-mp3")
def stream_mp3_audio_get(
    text: str = Query(..., min_length=1),
    language: str | None = None,
    ref_audio: str | None = "am_michael.mp3",
    ref_text: str | None = (
        "Human just went farther from earth than ever before. "
        "This was the mission to go back to the moon, with the goal of "
        "eventually establishing a moon colony."
    ),
    instruct: str | None = None,
    speed: float | None = 0.9,
    num_step: int = Query(default=16, ge=1),
    denoise: bool = False,
    postprocess_output: bool = False,
    chunk_chars: int = Query(default=120, ge=40, le=600),
) -> StreamingResponse:
    request = StreamRequest(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        instruct=instruct,
        speed=speed,
        num_step=num_step,
        denoise=denoise,
        postprocess_output=postprocess_output,
        chunk_chars=chunk_chars,
    )
    return stream_mp3_audio(request)
