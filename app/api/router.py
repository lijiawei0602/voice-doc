from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.tasks import router as tasks_router
from app.api.routes.transcriptions import router as transcriptions_router
from app.api.routes.streaming import router as streaming_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(transcriptions_router)
api_router.include_router(tasks_router)
api_router.include_router(streaming_router)
