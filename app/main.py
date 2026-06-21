from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import configure_logging, get_logger
from app.core.response import error_response

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="语音转文字后端服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.include_router(api_router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "服务启动完成: app=%s engine=%s port=%s",
        settings.app_name,
        settings.engine,
        settings.port,
    )


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.detail.http_status,
        content=error_response(exc.detail.code, exc.detail.message),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.error("未处理异常: %s\n%s", exc, traceback.format_exc())
    detail = ERRORS["INTERNAL_ERROR"]
    return JSONResponse(
        status_code=detail.http_status,
        content=error_response(detail.code, detail.message, {"reason": str(exc)}),
    )
