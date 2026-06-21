from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioChunk:
    path: Path
    index: int
    start_ms: int
    end_ms: int


def split_audio_if_needed(audio_path: Path, duration_seconds: float) -> list[AudioChunk]:
    settings = get_settings()
    chunk_seconds = settings.audio_chunk_seconds

    if duration_seconds <= chunk_seconds:
        return [
            AudioChunk(
                path=audio_path,
                index=0,
                start_ms=0,
                end_ms=int(duration_seconds * 1000),
            )
        ]

    logger.warning(
        "音频较长，启用分片处理: duration=%.2fs, chunk=%ss",
        duration_seconds,
        chunk_seconds,
    )
    output_dir = settings.temp_dir / f"chunks_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c",
            "copy",
            str(output_dir / "chunk_%03d.wav"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    chunks: list[AudioChunk] = []
    for index, chunk_path in enumerate(sorted(output_dir.glob("chunk_*.wav"))):
        start_ms = index * chunk_seconds * 1000
        end_ms = start_ms + chunk_seconds * 1000
        chunks.append(
            AudioChunk(
                path=chunk_path,
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return chunks
