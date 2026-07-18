#!/usr/bin/env python3
"""
模型预下载脚本

在服务启动前运行此脚本，提前下载并缓存模型到本地，
避免服务启动时在线拉取导致启动时间过长。

使用方法:
    python scripts/preload_models.py

或直接运行:
    uv run python scripts/preload_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def preload_models():
    """预加载所有 FunASR 模型"""
    from app.core.config import get_settings
    from app.core.logging import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info("=" * 50)
    logger.info("开始预下载 FunASR 模型")
    logger.info("=" * 50)
    logger.info("模型缓存目录: %s", settings.model_cache_dir)
    logger.info("Hub: %s", settings.funasr_hub)
    logger.info("设备: %s", "cuda" if not settings.force_cpu else "cpu")

    # 确保缓存目录存在
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from funasr import AutoModel

        # 1. 下载普通识别模型
        logger.info("-" * 50)
        logger.info("1. 下载普通识别模型: %s", settings.funasr_model)
        logger.info("   (包含 VAD、PUNC、SPK 模型)")

        model_kwargs = {
            "model": settings.funasr_model,
            "vad_model": settings.funasr_vad_model,
            "punc_model": settings.funasr_punc_model,
            "spk_model": settings.funasr_spk_model,
            "device": "cpu",  # 预下载使用 CPU
            "hub": settings.funasr_hub,
            "model_cache_dir": str(settings.model_cache_dir),
            "disable_update": True,
        }

        try:
            model = AutoModel(**model_kwargs)
            logger.info("   ✓ 普通识别模型下载完成")
        except TypeError:
            model_kwargs.pop("model_cache_dir", None)
            model = AutoModel(**model_kwargs)
            logger.info("   ✓ 普通识别模型下载完成")
        except Exception as e:
            logger.error("   ✗ 普通识别模型下载失败: %s", e)

        # 2. 下载流式识别专用模型
        logger.info("-" * 50)
        logger.info("2. 下载流式识别模型: %s", settings.funasr_streaming_model)

        streaming_kwargs = {
            "model": settings.funasr_streaming_model,
            "device": "cpu",  # 预下载使用 CPU
            "hub": settings.funasr_hub,
            "model_cache_dir": str(settings.model_cache_dir),
            "disable_update": True,
        }

        try:
            streaming_model = AutoModel(**streaming_kwargs)
            logger.info("   ✓ 流式识别模型下载完成")
        except TypeError:
            streaming_kwargs.pop("model_cache_dir", None)
            streaming_model = AutoModel(**streaming_kwargs)
            logger.info("   ✓ 流式识别模型下载完成")
        except Exception as e:
            logger.error("   ✗ 流式识别模型下载失败: %s", e)

        # 列出已缓存的模型
        logger.info("-" * 50)
        logger.info("已缓存的模型:")
        if settings.model_cache_dir.exists():
            for model_dir in settings.model_cache_dir.iterdir():
                if model_dir.is_dir():
                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    size_mb = size / (1024 * 1024)
                    logger.info("   - %s (%.1f MB)", model_dir.name, size_mb)
        else:
            logger.info("   (无)")

        logger.info("=" * 50)
        logger.info("模型预下载完成!")
        logger.info("=" * 50)

    except ImportError as e:
        logger.error("导入 FunASR 失败: %s", e)
        logger.error("请先安装 FunASR: pip install funasr")
        sys.exit(1)
    except Exception as e:
        logger.error("预下载过程中出错: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    preload_models()
