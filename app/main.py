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
        "服务启动: app=%s engine=%s port=%s",
        settings.app_name,
        settings.engine,
        settings.port,
    )
    
    # 预加载模型
    if settings.preload_models_on_startup:
        logger.info("开始预加载模型...")
        try:
            from app.services.asr.funasr_engine import check_local_model_cache, get_funasr_engine
            
            cache_dir = settings.model_cache_dir
            
            # 检查普通识别模型缓存
            has_normal_cache = check_local_model_cache(settings.funasr_model, cache_dir)
            cache_status = "本地缓存" if has_normal_cache else "在线加载"
            logger.info("  - 加载普通识别模型 [%s] (%s)...", settings.funasr_model, cache_status)
            logger.info("    - spk_model: %s", settings.funasr_spk_model)
            logger.info("    - punc_model: %s", settings.funasr_punc_model)
            engine = get_funasr_engine()
            logger.info("  ✓ 普通识别模型加载完成")
            
            # 检查流式识别模型缓存
            has_streaming_cache = check_local_model_cache(settings.funasr_streaming_model, cache_dir)
            cache_status = "本地缓存" if has_streaming_cache else "在线加载"
            logger.info("  - 加载流式识别模型 [%s] (%s)...", settings.funasr_streaming_model, cache_status)
            engine.load_streaming_model()
            logger.info("  ✓ 流式识别模型加载完成")
            
            logger.info("模型预加载完成!")
        except Exception as exc:
            logger.warning("模型预加载失败，将在首次请求时加载: %s", exc)
    
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
