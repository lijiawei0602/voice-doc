from __future__ import annotations

from app.core.config import get_settings


def detect_device() -> str:
    settings = get_settings()
    if settings.force_cpu or not settings.enable_gpu:
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        return "cpu"

    return "cpu"
