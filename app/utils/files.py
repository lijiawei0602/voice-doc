from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import raise_app_error


def validate_audio_extension(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in get_settings().supported_audio_extensions:
        raise_app_error("INVALID_AUDIO_FORMAT")
    return suffix


async def save_upload_file(upload_file: UploadFile) -> Path:
    settings = get_settings()
    suffix = validate_audio_extension(upload_file.filename or "")
    target_path = settings.temp_dir / f"upload_{uuid4().hex}{suffix}"

    size_bytes = 0
    with target_path.open("wb") as output:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_size_mb * 1024 * 1024:
                target_path.unlink(missing_ok=True)
                raise_app_error("FILE_TOO_LARGE")
            output.write(chunk)

    await upload_file.close()
    return target_path


def stage_local_file(file_path: str) -> Path:
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"文件不存在: {source}")

    validate_audio_extension(source.name)
    settings = get_settings()
    target = settings.temp_dir / f"local_{uuid4().hex}{source.suffix.lower()}"
    shutil.copy2(source, target)
    return target
