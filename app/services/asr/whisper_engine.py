from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.models.schemas import TranscriptSegment
from app.services.asr.base import BaseAsrEngine, EngineResult
from app.services.diarization.pyannote_diarizer import PyannoteDiarizer, SpeakerTurn
from app.utils.device import detect_device

logger = get_logger(__name__)


class WhisperEngine(BaseAsrEngine):
    engine_name = "whisper"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = detect_device()
        self.model: Optional[Any] = None
        self.diarizer = PyannoteDiarizer()

    def load(self) -> None:
        if self.model is not None:
            return

        try:
            import whisper
        except Exception as exc:
            logger.error("Whisper 导入失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        try:
            self.model = whisper.load_model(
                self.settings.whisper_model,
                download_root=str(self.settings.whisper_download_root),
                device="cuda" if self.device.startswith("cuda") else "cpu",
            )
            logger.info("Whisper 模型加载完成，device=%s", self.device)
        except Exception as exc:
            logger.error("Whisper 模型加载失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

    def transcribe(self, audio_path: Path) -> EngineResult:
        self.load()

        try:
            raw = self.model.transcribe(
                str(audio_path),
                language=self.settings.whisper_language,
                task="transcribe",
                word_timestamps=True,
                verbose=False,
                fp16=self.device.startswith("cuda"),
            )
        except MemoryError as exc:
            raise AppError(ERRORS["MEMORY_OVERFLOW"]) from exc
        except Exception as exc:
            logger.error("Whisper 转写失败: %s", exc)
            raise AppError(ERRORS["TRANSCRIPTION_FAILED"]) from exc

        speaker_turns = self.diarizer.diarize(audio_path)
        segments = self._merge_segments(raw.get("segments", []), speaker_turns)
        full_text = " ".join(segment.text for segment in segments).strip()

        return EngineResult(
            text=full_text,
            segments=segments,
            language=raw.get("language"),
            metadata={"speaker_turns": [turn.__dict__ for turn in speaker_turns]},
        )

    def _merge_segments(
        self,
        whisper_segments: list[dict[str, Any]],
        speaker_turns: list[SpeakerTurn],
    ) -> list[TranscriptSegment]:
        merged: list[TranscriptSegment] = []
        for segment in whisper_segments:
            start_ms = int(float(segment.get("start", 0)) * 1000)
            end_ms = int(float(segment.get("end", 0)) * 1000)
            speaker = self._match_speaker(start_ms, end_ms, speaker_turns)
            merged.append(
                TranscriptSegment(
                    speaker=speaker,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=str(segment.get("text", "")).strip(),
                )
            )
        return merged

    @staticmethod
    def _match_speaker(
        start_ms: int,
        end_ms: int,
        speaker_turns: list[SpeakerTurn],
    ) -> str:
        best_speaker = "spk0"
        best_overlap = -1

        for turn in speaker_turns:
            overlap = min(end_ms, turn.end_ms) - max(start_ms, turn.start_ms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        return best_speaker
