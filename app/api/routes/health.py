from fastapi import APIRouter

from app.core.config import get_settings
from app.core.response import success_response
from app.utils.device import detect_device

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    settings = get_settings()
    return success_response(
        {
            "app": settings.app_name,
            "engine": settings.engine,
            "device": detect_device(),
        }
    )
