from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.models.schemas import StreamingSegment, StreamingResult
from app.models.schemas import TranscriptSegment
from app.services.asr.base import BaseAsrEngine, EngineResult
from app.utils.device import detect_device

logger = get_logger(__name__)

# 流式识别默认参数（参考 FunASR 官方示例）
DEFAULT_CHUNK_SIZE = [0, 10, 5]  # 600ms 显示，300ms 前瞻
DEFAULT_SAMPLE_RATE = 16000

# 模块级单例（类变量）
_instance: Optional["FunAsrEngine"] = None
_model_loaded: bool = False


def get_funasr_engine() -> "FunAsrEngine":
    """获取 FunAsrEngine 单例实例（线程安全）"""
    global _instance
    if _instance is None:
        _instance = FunAsrEngine()
        _instance.load()
    return _instance


class FunAsrEngine(BaseAsrEngine):
    engine_name = "funasr"

    # 类级别的模型实例，所有实例共享
    _shared_model: Optional[Any] = None
    _model_initialized: bool = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = detect_device()
        self._streaming_sessions: dict[str, dict] = {}

    @property
    def model(self) -> Any:
        """延迟加载模型，使用类级别共享实例"""
        return self._shared_model

    def load(self) -> None:
        """加载模型（类级别，只加载一次）"""
        if FunAsrEngine._model_initialized:
            logger.debug("FunASR 模型已加载，跳过")
            return

        try:
            from funasr import AutoModel
        except Exception as exc:
            logger.error("FunASR 导入失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        model_kwargs = {
            "model": self.settings.funasr_model,
            "vad_model": self.settings.funasr_vad_model,
            "punc_model": self.settings.funasr_punc_model,
            "spk_model": self.settings.funasr_spk_model,
            "device": self.device,
            "hub": self.settings.funasr_hub,
            "model_cache_dir": str(self.settings.model_cache_dir),
            "disable_update": True, # 新增此行关闭版本更新检查
        }
        try:
            try:
                FunAsrEngine._shared_model = AutoModel(**model_kwargs)
            except TypeError:
                model_kwargs.pop("model_cache_dir", None)
                FunAsrEngine._shared_model = AutoModel(**model_kwargs)
            FunAsrEngine._model_initialized = True
            logger.info("FunASR 模型加载完成，device=%s", self.device)
        except Exception as exc:
            logger.error("FunASR 模型加载失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

    def create_streaming_session(
        self,
        chunk_size: list[int] | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> str:
        """创建流式识别会话

        Args:
            chunk_size:  chunk 大小配置 [left_chunks, middle_chunks, right_chunks]
                        默认 [0, 10, 5]，表示 600ms 显示，300ms 前瞻
            sample_rate: 采样率，默认 16000
        """
        self.load()
        session_id = uuid4().hex
        
        # 验证 chunk_size 类型
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE
        
        # 确保是有效列表且有足够元素
        if not isinstance(chunk_size, (list, tuple)) or len(chunk_size) < 2:
            logger.warning("无效的 chunk_size: %s，使用默认值", chunk_size)
            chunk_size = DEFAULT_CHUNK_SIZE
        
        # 确保元素是整数
        try:
            chunk_size = [int(x) for x in chunk_size[:3]]  # 只取前3个元素
        except (ValueError, TypeError) as e:
            logger.warning("chunk_size 包含非整数元素: %s，使用默认值", chunk_size)
            chunk_size = DEFAULT_CHUNK_SIZE

        self._streaming_sessions[session_id] = {
            "session_id": session_id,
            "cache": {},
            "segment_id": 0,
            "full_text": "",
            "chunk_size": chunk_size,
            "chunk_stride": chunk_size[1] * sample_rate // 10,  # 每块采样点数
            "sample_rate": sample_rate,
            "last_text": "",  # 用于增量输出
        }
        logger.info("创建流式会话: session_id=%s, chunk_size=%s", session_id, chunk_size)
        return session_id

    def close_streaming_session(self, session_id: str) -> None:
        """关闭流式识别会话"""
        if session_id in self._streaming_sessions:
            del self._streaming_sessions[session_id]
            logger.info("关闭流式会话: session_id=%s", session_id)

    async def stream_transcribe(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        is_final: bool = False,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
    ) -> StreamingSegment | None:
        """处理单个音频块进行流式识别

        Args:
            session_id: 会话ID
            audio_chunk: numpy 音频数组（float32，16kHz）
            is_final: 是否为最后一个音频块
            encoder_chunk_look_back: encoder 回看块数
            decoder_chunk_look_back: decoder 回看块数

        Returns:
            StreamingSegment: 识别片段（仅当有新文本时返回）
        """
        if session_id not in self._streaming_sessions:
            session_id = self.create_streaming_session()

        session = self._streaming_sessions[session_id]
        
        # 确保 chunk_size 是正确的列表类型
        chunk_size = session.get("chunk_size", DEFAULT_CHUNK_SIZE)
        if isinstance(chunk_size, str):
            try:
                import json as json_lib
                chunk_size = json_lib.loads(chunk_size)
            except Exception:
                chunk_size = DEFAULT_CHUNK_SIZE
        if not isinstance(chunk_size, (list, tuple)) or len(chunk_size) < 2:
            chunk_size = DEFAULT_CHUNK_SIZE
        try:
            chunk_size = [int(x) for x in chunk_size[:3]]
        except (ValueError, TypeError):
            chunk_size = DEFAULT_CHUNK_SIZE

        try:
            if len(audio_chunk) < 1600:  # 音频太短，忽略
                return None

            generate_kwargs = {
                "input": audio_chunk,
                "cache": session["cache"],
                "is_final": is_final,
                "chunk_size": chunk_size,
                "encoder_chunk_look_back": encoder_chunk_look_back,
                "decoder_chunk_look_back": decoder_chunk_look_back,
            }

            raw = await asyncio.to_thread(
                self.model.generate, **generate_kwargs
            )

            result = raw[0] if isinstance(raw, list) else raw
            session["cache"] = result.get("cache", {})

            text = str(result.get("text", "")).strip()
            if not text:
                return None

            # 计算增量文本
            incremental_text = text
            if not is_final and text.startswith(session["last_text"]):
                incremental_text = text[len(session["last_text"]):]
            session["last_text"] = text

            segment = StreamingSegment(
                segment_id=session["segment_id"],
                text=incremental_text,
                start_ms=0,
                end_ms=0,
                is_final=is_final,
                speaker="spk0",
            )
            session["segment_id"] += 1
            session["full_text"] += incremental_text

            return segment

        except Exception as exc:
            import traceback
            logger.error("流式识别失败: session_id=%s, error=%s", session_id, exc)
            logger.error("chunk_size 类型: %s, 值: %s", type(chunk_size), chunk_size)
            logger.error("音频块长度: %s", len(audio_chunk) if audio_chunk is not None else "None")
            logger.error("generate_kwargs: %s", generate_kwargs)
            logger.error("异常堆栈: %s", traceback.format_exc())
            raise AppError(ERRORS["TRANSCRIPTION_FAILED"]) from exc

    async def stream_audio(
        self,
        session_id: str,
        audio_data: np.ndarray,
        is_final: bool = False,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
    ) -> list[StreamingSegment]:
        """对完整音频数据进行分块流式识别

        Args:
            session_id: 会话ID
            audio_data: 完整音频数据（float32，16kHz）
            is_final: 是否为最后一个音频块
            encoder_chunk_look_back: encoder 回看块数
            decoder_chunk_look_back: decoder 回看块数

        Returns:
            list[StreamingSegment]: 所有识别片段
        """
        if session_id not in self._streaming_sessions:
            session_id = self.create_streaming_session()

        session = self._streaming_sessions[session_id]
        chunk_stride = session["chunk_stride"]

        segments = []
        total_chunks = int((len(audio_data) - 1) / chunk_stride + 1)

        for i in range(total_chunks):
            chunk = audio_data[i * chunk_stride:(i + 1) * chunk_stride]
            chunk_is_final = is_final or (i == total_chunks - 1)

            segment = await self.stream_transcribe(
                session_id=session_id,
                audio_chunk=chunk,
                is_final=chunk_is_final,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )

            if segment:
                segments.append(segment)

        return segments

    async def stream_transcribe_bytes(
        self,
        session_id: str,
        audio_bytes: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        is_final: bool = False,
    ) -> StreamingResult:
        """从字节数据流式识别

        Args:
            session_id: 会话ID
            audio_bytes: PCM 音频数据（16位整数）
            sample_rate: 采样率
            is_final: 是否为最后一个音频块

        Returns:
            StreamingResult: 流式识别结果
        """
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        segments = await self.stream_audio(
            session_id=session_id,
            audio_data=audio_float,
            is_final=is_final,
        )

        session = self._streaming_sessions.get(session_id, {})
        return StreamingResult(
            session_id=session_id,
            text=session.get("full_text", ""),
            segments=segments,
        )

    def transcribe(self, audio_path: Path) -> EngineResult:
        self.load()

        try:
            raw = self.model.generate(
                input=str(audio_path),
                batch_size_s=self.settings.batch_size_seconds,
            )
        except MemoryError as exc:
            raise AppError(ERRORS["MEMORY_OVERFLOW"]) from exc
        except Exception as exc:
            logger.error("FunASR 推理失败: %s", exc)
            raise AppError(ERRORS["TRANSCRIPTION_FAILED"]) from exc

        # 处理空结果
        if isinstance(raw, list):
            if len(raw) == 0:
                logger.warning("FunASR 返回空结果: %s", audio_path)
                return EngineResult(
                    text="",
                    segments=[],
                    language=None,
                    metadata={"raw_sentence_count": 0},
                )
            result = raw[0]
        else:
            result = raw
        sentence_info = result.get("sentence_info", []) or []

        segments: list[TranscriptSegment] = []
        if sentence_info:
            for item in sentence_info:
                segments.append(
                    TranscriptSegment(
                        speaker=self._normalize_speaker(item.get("spk")),
                        start_ms=int(item.get("start", 0)),
                        end_ms=int(item.get("end", 0)),
                        text=str(item.get("text", "")).strip(),
                    )
                )
        else:
            segments.append(
                TranscriptSegment(
                    speaker="spk0",
                    start_ms=0,
                    end_ms=0,
                    text=str(result.get("text", "")).strip(),
                )
            )

        return EngineResult(
            text=str(result.get("text", "")).strip(),
            segments=segments,
            language=result.get("language"),
            metadata={
                "raw_sentence_count": len(sentence_info),
                "timestamps": result.get("timestamp"),
            },
        )

    @staticmethod
    def _normalize_speaker(value: Any) -> str:
        if value is None:
            return "spk0"
        text = str(value).lower().replace("speaker", "").replace("_", "")
        digits = "".join(ch for ch in text if ch.isdigit())
        return f"spk{digits or '0'}"
