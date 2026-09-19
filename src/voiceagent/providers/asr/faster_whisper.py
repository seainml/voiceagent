"""faster-whisper — local, offline transcription (optional ``local-asr`` extra).

The model is loaded lazily and the blocking call is pushed to a thread so the
event loop keeps servicing the WebSocket while a sentence is being decoded.
``hotwords`` matters more than model size in practice: pinning your colleagues'
names, project codenames and CLI names into the prompt fixes most of the
errors that actually annoy people.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...audio.pcm import TARGET_SAMPLE_RATE, AudioClip
from ...audio.vad import VADConfig, trim_silence
from ...config import ASRSettings
from ..base import ProviderError, ProviderUnavailable, Transcript


class FasterWhisperASR:
    name = "faster_whisper"

    def __init__(self, settings: ASRSettings) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderUnavailable(
                "faster-whisper is not installed; run: uv pip install -e '.[local-asr]'"
            ) from exc
        self._model_cls = WhisperModel
        self.settings = settings
        self._model: Any = None
        self._lock = asyncio.Lock()

    # -- lifecycle -------------------------------------------------------

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
        return self._model

    def _load(self) -> Any:
        root = self.settings.download_root
        if root:
            Path(root).expanduser().mkdir(parents=True, exist_ok=True)
        try:
            return self._model_cls(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
                download_root=str(Path(root).expanduser()) if root else None,
            )
        except Exception as exc:  # pragma: no cover - model download is heavy
            raise ProviderError(
                f"could not load faster-whisper model '{self.settings.model}': {exc}"
            ) from exc

    # -- provider API ----------------------------------------------------

    async def warmup(self) -> None:
        """Load the model ahead of the first turn.

        Without this the first utterance pays the whole model-load cost — a
        multi-second stall right when the user is judging whether the thing
        works at all.
        """
        await self._ensure_model()

    async def transcribe(
        self,
        clip: AudioClip,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> Transcript:
        if clip.n_samples == 0:
            return Transcript(text="", provider=self.name)
        model = await self._ensure_model()
        audio = clip.resample(TARGET_SAMPLE_RATE)
        trimmed = trim_silence(audio, VADConfig(), pad_ms=120)
        if trimmed.n_samples == 0:
            return Transcript(text="", duration_s=clip.duration_s, provider=self.name)
        samples = trimmed.samples

        prompt = ", ".join(hotwords) if hotwords else None
        try:
            text, info = await asyncio.to_thread(self._run, model, samples, language, prompt)
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise ProviderError(f"transcription failed: {exc}") from exc
        return Transcript(
            text=text,
            language=getattr(info, "language", language),
            duration_s=clip.duration_s,
            confidence=getattr(info, "language_probability", None),
            provider=self.name,
        )

    def _run(
        self,
        model: Any,
        samples: Any,
        language: str | None,
        prompt: str | None,
    ) -> tuple[str, Any]:
        segments, info = model.transcribe(
            samples,
            language=language or self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip(), info

    async def aclose(self) -> None:
        self._model = None


__all__ = ["FasterWhisperASR"]
