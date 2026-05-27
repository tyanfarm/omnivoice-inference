from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from omnivoice import OmniVoice
from pydantic import BaseModel, Field
from voices import VOICE_METADATA

try:
    import lameenc
except ImportError:
    lameenc = None

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

app = FastAPI(title="OmniVoice Streaming Test API")

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
    def _audio_chunk_is_empty(audio: torch.Tensor | np.ndarray | None) -> bool:
        if audio is None:
            return True
        if isinstance(audio, torch.Tensor):
            return audio.numel() == 0
        return np.asarray(audio).size == 0

    @classmethod
    def _audio_tensor_to_int16_bytes(cls, audio: torch.Tensor | np.ndarray) -> bytes:
        chunk_np = cls._audio_to_numpy(audio)
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

        ref_audio_path, ref_text = self.get_voice_clone_config(request.voice_id)

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
            if ref_audio_path:
                voice_clone_prompt = self._get_voice_clone_prompt(
                    str(ref_audio_path),
                    ref_text,
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
                if self._audio_chunk_is_empty(audio_chunk):
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
