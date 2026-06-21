from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.schemas import TranscriptSegment


@dataclass
class EngineResult:
    text: str
    segments: list[TranscriptSegment]
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAsrEngine(ABC):
    engine_name: str

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: Path) -> EngineResult:
        raise NotImplementedError
