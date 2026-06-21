from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS, raise_app_error
from app.core.logging import get_logger

logger = get_logger(__name__)


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        logger.error("命令执行失败: %s | stderr=%s", " ".join(command), exc.stderr)
        raise AppError(ERRORS["AUDIO_CORRUPTED"]) from exc


def probe_duration_seconds(audio_path: Path) -> float:
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )
    try:
        return float(output)
    except ValueError as exc:
        raise AppError(ERRORS["AUDIO_CORRUPTED"]) from exc


def preprocess_audio(input_path: Path) -> tuple[Path, float]:
    settings = get_settings()
    duration = probe_duration_seconds(input_path)
    if duration > settings.max_audio_duration_seconds:
        raise_app_error("AUDIO_TOO_LONG")

    output_path = settings.temp_dir / f"preprocessed_{uuid4().hex}.wav"
    filter_chain = "highpass=f=200,lowpass=f=3000"
    if settings.denoise_enabled:
        filter_chain += ",afftdn=nf=-25"

    logger.info("开始音频预处理: %s", input_path)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(settings.sample_rate),
            "-af",
            filter_chain,
            str(output_path),
        ]
    )
    return output_path, duration
