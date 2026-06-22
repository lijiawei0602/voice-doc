from __future__ import annotations

import traceback
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.models.schemas import SpeakerSummary, TranscriptResult, TranscriptSegment
from app.services.asr.base import BaseAsrEngine, EngineResult
from app.services.asr.funasr_engine import FunAsrEngine
from app.services.asr.whisper_engine import WhisperEngine
from app.services.audio.chunking import AudioChunk, split_audio_if_needed
from app.services.audio.preprocess import preprocess_audio
from app.utils.device import detect_device
from app.utils.json_store import write_json

logger = get_logger(__name__)


class TranscriptionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = self._build_engine()
        self.device = detect_device()

    def transcribe_file(self, source_path: Path, original_source: str | None = None) -> TranscriptResult:
        task_id = uuid4().hex
        return self.transcribe_file_with_id(
            source_path=source_path,
            task_id=task_id,
            original_source=original_source or str(source_path),
        )

    def transcribe_file_with_id(
        self,
        source_path: Path,
        task_id: str,
        original_source: str,
    ) -> TranscriptResult:
        try:
            normalized_path, duration_seconds = preprocess_audio(source_path)
            chunks = split_audio_if_needed(normalized_path, duration_seconds)
            merged_segments: list[TranscriptSegment] = []
            metadata: dict[str, object] = {
                "chunk_count": len(chunks),
                "chunk_seconds": self.settings.audio_chunk_seconds,
            }

            for chunk in chunks:
                logger.info("处理分片: task_id=%s chunk=%s path=%s", task_id, chunk.index, chunk.path)
                chunk_result = self.engine.transcribe(chunk.path)
                metadata[f"chunk_{chunk.index}"] = {
                    "language": chunk_result.language,
                    **chunk_result.metadata,
                }
                merged_segments.extend(self._apply_chunk_offset(chunk_result, chunk))

            merged_segments = self._normalize_speakers(merged_segments)
            full_text = " ".join(segment.text for segment in merged_segments).strip()
            speaker_summaries = self._build_speaker_summaries(merged_segments)

            result_path = self.settings.result_dir / f"{task_id}.json"
            result = TranscriptResult(
                task_id=task_id,
                engine=self.settings.engine,
                device=self.device,
                source=original_source,
                language=self._first_chunk_language(metadata),
                duration_seconds=round(duration_seconds, 3),
                text=full_text,
                segments=merged_segments,
                speakers=speaker_summaries,
                result_path=result_path,
                metadata=metadata,
            )
            write_json(result_path, result.model_dump(mode="json"))
            return result
        except AppError:
            raise
        except MemoryError as exc:
            raise AppError(ERRORS["MEMORY_OVERFLOW"]) from exc
        except Exception as exc:
            logger.error("识别流程异常: %s\n%s", exc, traceback.format_exc())
            raise AppError(ERRORS["TRANSCRIPTION_FAILED"]) from exc

    def _build_engine(self) -> BaseAsrEngine:
        if self.settings.engine == "funasr":
            return FunAsrEngine()
        if self.settings.engine == "whisper":
            return WhisperEngine()
        raise AppError(ERRORS["INTERNAL_ERROR"])

    @staticmethod
    def _apply_chunk_offset(result: EngineResult, chunk: AudioChunk) -> list[TranscriptSegment]:
        offset_segments: list[TranscriptSegment] = []
        for segment in result.segments:
            offset_segments.append(
                TranscriptSegment(
                    speaker=segment.speaker,
                    start_ms=segment.start_ms + chunk.start_ms,
                    end_ms=segment.end_ms + chunk.start_ms,
                    text=segment.text,
                    chunk_index=chunk.index,
                )
            )
        return offset_segments

    @staticmethod
    def _normalize_speakers(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        speaker_map: dict[str, str] = {}
        normalized: list[TranscriptSegment] = []
        for segment in segments:
            speaker = speaker_map.setdefault(segment.speaker, f"spk{len(speaker_map)}")
            normalized.append(
                TranscriptSegment(
                    speaker=speaker,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text,
                    chunk_index=segment.chunk_index,
                )
            )
        return normalized

    @staticmethod
    def _build_speaker_summaries(segments: list[TranscriptSegment]) -> list[SpeakerSummary]:
        grouped: dict[str, list[TranscriptSegment]] = {}
        for segment in segments:
            grouped.setdefault(segment.speaker, []).append(segment)

        speakers: list[SpeakerSummary] = []
        for speaker, speaker_segments in grouped.items():
            text = " ".join(item.text for item in speaker_segments).strip()
            speakers.append(
                SpeakerSummary(
                    speaker=speaker,
                    text=text,
                    segments=speaker_segments,
                )
            )
        return speakers

    @staticmethod
    def _first_chunk_language(metadata: dict[str, object]) -> Optional[str]:
        first_chunk = metadata.get("chunk_0")
        if isinstance(first_chunk, dict):
            language = first_chunk.get("language")
            if isinstance(language, str):
                return language
        return None
