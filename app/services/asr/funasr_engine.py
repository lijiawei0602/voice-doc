from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.models.schemas import TranscriptSegment
from app.services.asr.base import BaseAsrEngine, EngineResult
from app.utils.device import detect_device

logger = get_logger(__name__)


class FunAsrEngine(BaseAsrEngine):
    engine_name = "funasr"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = detect_device()
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        try:
            from funasr import AutoModel
        except Exception as exc:
            logger.error("FunASR 导入失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        model_kwargs = {
            "model": self.settings.funasr_model,
            "vad_model": self.settings.funasr_vad_model,
            "punc_model": self.settings.funasr_punc_model,
            "spk_model": self.settings.funasr_spk_model,
            "device": self.device,
            "hub": self.settings.funasr_hub,
            "model_cache_dir": str(self.settings.model_cache_dir),
        }
        try:
            try:
                self.model = AutoModel(**model_kwargs)
            except TypeError:
                model_kwargs.pop("model_cache_dir", None)
                self.model = AutoModel(**model_kwargs)
            logger.info("FunASR 模型加载完成，device=%s", self.device)
        except Exception as exc:
            logger.error("FunASR 模型加载失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

    def transcribe(self, audio_path: Path) -> EngineResult:
        self.load()

        try:
            raw = self.model.generate(
                input=str(audio_path),
                batch_size_s=self.settings.batch_size_seconds,
            )
        except MemoryError as exc:
            raise AppError(ERRORS["MEMORY_OVERFLOW"]) from exc
        except Exception as exc:
            logger.error("FunASR 推理失败: %s", exc)
            raise AppError(ERRORS["TRANSCRIPTION_FAILED"]) from exc

        result = raw[0] if isinstance(raw, list) else raw
        sentence_info = result.get("sentence_info", []) or []

        segments: list[TranscriptSegment] = []
        if sentence_info:
            for item in sentence_info:
                segments.append(
                    TranscriptSegment(
                        speaker=self._normalize_speaker(item.get("spk")),
                        start_ms=int(item.get("start", 0)),
                        end_ms=int(item.get("end", 0)),
                        text=str(item.get("text", "")).strip(),
                    )
                )
        else:
            segments.append(
                TranscriptSegment(
                    speaker="spk0",
                    start_ms=0,
                    end_ms=0,
                    text=str(result.get("text", "")).strip(),
                )
            )

        return EngineResult(
            text=str(result.get("text", "")).strip(),
            segments=segments,
            language=result.get("language"),
            metadata={
                "raw_sentence_count": len(sentence_info),
                "timestamps": result.get("timestamp"),
            },
        )

    @staticmethod
    def _normalize_speaker(value: Any) -> str:
        if value is None:
            return "spk0"
        text = str(value).lower().replace("speaker", "").replace("_", "")
        digits = "".join(ch for ch in text if ch.isdigit())
        return f"spk{digits or '0'}"
