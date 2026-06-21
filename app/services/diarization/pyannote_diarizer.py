from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.utils.device import detect_device

logger = get_logger(__name__)


@dataclass
class SpeakerTurn:
    speaker: str
    start_ms: int
    end_ms: int


class PyannoteDiarizer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = detect_device()
        self.pipeline = None

    def load(self) -> None:
        if self.pipeline is not None:
            return

        try:
            import torch
            from pyannote.audio import Pipeline
        except Exception as exc:
            logger.error("pyannote.audio 导入失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        try:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.settings.pyannote_model,
                    use_auth_token=self.settings.pyannote_auth_token,
                    cache_dir=str(self.settings.model_cache_dir),
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    self.settings.pyannote_model,
                    use_auth_token=self.settings.pyannote_auth_token,
                )
            if self.device.startswith("cuda"):
                self.pipeline.to(torch.device("cuda"))
            logger.info("pyannote 说话人分离模型加载完成，device=%s", self.device)
        except Exception as exc:
            logger.error("pyannote 模型加载失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        self.load()
        try:
            call_kwargs = {}
            if self.settings.pyannote_num_speakers:
                call_kwargs["num_speakers"] = self.settings.pyannote_num_speakers
            diarization = self.pipeline(str(audio_path), **call_kwargs)
        except Exception as exc:
            logger.error("pyannote 说话人分离失败: %s", exc)
            raise AppError(ERRORS["DIARIZATION_FAILED"]) from exc

        speaker_map: dict[str, str] = {}
        turns: list[SpeakerTurn] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            normalized = speaker_map.setdefault(speaker, f"spk{len(speaker_map)}")
            turns.append(
                SpeakerTurn(
                    speaker=normalized,
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                )
            )
        return turns
