from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ErrorDetail:
    code: str
    message: str
    http_status: int


class AppError(Exception):
    def __init__(self, detail: ErrorDetail):
        super().__init__(detail.message)
        self.detail = detail


ERRORS = {
    "INVALID_AUDIO_FORMAT": ErrorDetail(
        code="INVALID_AUDIO_FORMAT",
        message="仅支持 mp3/wav/flac/m4a 音频格式。",
        http_status=400,
    ),
    "FILE_TOO_LARGE": ErrorDetail(
        code="FILE_TOO_LARGE",
        message="上传文件超过系统限制。",
        http_status=413,
    ),
    "AUDIO_TOO_LONG": ErrorDetail(
        code="AUDIO_TOO_LONG",
        message="音频时长超过系统限制。",
        http_status=400,
    ),
    "AUDIO_CORRUPTED": ErrorDetail(
        code="AUDIO_CORRUPTED",
        message="音频损坏或无法解码。",
        http_status=400,
    ),
    "MODEL_LOAD_FAILED": ErrorDetail(
        code="MODEL_LOAD_FAILED",
        message="模型加载失败，请检查依赖、模型路径或显存配置。",
        http_status=500,
    ),
    "TRANSCRIPTION_FAILED": ErrorDetail(
        code="TRANSCRIPTION_FAILED",
        message="音频识别失败。",
        http_status=500,
    ),
    "DIARIZATION_FAILED": ErrorDetail(
        code="DIARIZATION_FAILED",
        message="说话人分离失败。",
        http_status=500,
    ),
    "TASK_NOT_FOUND": ErrorDetail(
        code="TASK_NOT_FOUND",
        message="任务不存在。",
        http_status=404,
    ),
    "MEMORY_OVERFLOW": ErrorDetail(
        code="MEMORY_OVERFLOW",
        message="处理时内存不足，请缩短音频或减小并发。",
        http_status=500,
    ),
    "INTERNAL_ERROR": ErrorDetail(
        code="INTERNAL_ERROR",
        message="服务内部异常。",
        http_status=500,
    ),
}


def raise_app_error(code: str) -> None:
    raise AppError(ERRORS[code])
