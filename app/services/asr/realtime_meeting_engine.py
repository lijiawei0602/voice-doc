"""
实时会议纪要流式双路识别引擎
==========================

基于 FunASR paraformer-zh-streaming 的 1-pass + 2-pass 双路识别方案

架构设计：
1. VAD 静音检测 → 断句触发
2. Paraformer 流式 1-pass → 实时滚动字幕展示
3. 满足条件触发 2-pass → 全局修正结果作为最终纪要存档

2-pass 触发时机（会议场景标准）：
① VAD 连续静音超时（默认 0.5s）→ 自然断句
② 单人连续说话最长 6s → 防止长语音缓存漂移
③ 会议结束兜底 → 处理剩余缓存

Author: Voice-Doc Team
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.models.schemas import StreamingSegment, TranscriptSegment
from app.services.asr.base import BaseAsrEngine, EngineResult
from app.utils.device import detect_device

logger = get_logger(__name__)


# ==============================================================================
# 核心参数配置（FunASR 官方推荐）
# ==============================================================================

# 流式识别 chunk 配置：[left_chunks, middle_chunks, right_chunks]
# - left_chunks=0: 不回看左侧历史
# - middle_chunks=10: 每块 600ms (10 * 60ms)
# - right_chunks=5: 前瞻 300ms (5 * 60ms)
CHUNK_SIZE = [0, 10, 5]

# 音频参数
DEFAULT_SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = CHUNK_SIZE[1] * DEFAULT_SAMPLE_RATE // 10  # 16000 samples = 1s

# 2-pass 触发参数
VAD_SILENCE_TIMEOUT_MS = 500  # 静音超时 0.5s 触发 2-pass
MAX_SPEECH_DURATION_MS = 6000  # 单人连续说话超过 6s 强制分片 2-pass

# 模型路径
MODEL_NAME = "paraformer-zh-streaming"
VAD_MODEL_NAME = "fsmn-vad"


# ==============================================================================
# 状态枚举
# ==============================================================================

class SessionState(Enum):
    """会话状态枚举"""
    IDLE = "idle"  # 空闲状态
    SPEECH = "speech"  # 语音进行中
    SILENCE = "silence"  # 静音等待中


class TwoPassTrigger(Enum):
    """2-pass 触发原因枚举"""
    VAD_SILENCE = "vad_silence"  # VAD 静音超时触发
    LONG_SPEECH = "long_speech"  # 长语音强制分片触发
    SESSION_END = "session_end"  # 会话结束兜底触发


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class RealtimeSessionState:
    """实时会议会话状态（封装所有全局状态变量）"""

    # 会话基本信息
    session_id: str
    created_at: float = field(default_factory=time.time)

    # Chunk 配置
    chunk_size: list = field(default_factory=lambda: CHUNK_SIZE.copy())
    sample_rate: int = DEFAULT_SAMPLE_RATE

    # --------------------------------------------------------------------------
    # ASR 相关状态
    # --------------------------------------------------------------------------
    asr_cache: dict = field(default_factory=dict)  # ASR 推理缓存
    vad_cache: dict = field(default_factory=dict)  # VAD 推理缓存
    current_1pass_text: str = ""  # 当前 1-pass 实时文本（用于展示）
    last_1pass_text: str = ""  # 上一次 1-pass 文本（用于增量计算）

    # --------------------------------------------------------------------------
    # VAD 静音检测状态
    # --------------------------------------------------------------------------
    session_state: SessionState = SessionState.IDLE
    speech_start_time: Optional[float] = None  # 当前语音段开始时间
    silence_start_time: Optional[float] = None  # 当前静音段开始时间
    total_speech_duration_ms: float = 0  # 当前连续语音总时长

    # --------------------------------------------------------------------------
    # 音频缓存（用于 2-pass 最终识别）
    # --------------------------------------------------------------------------
    pending_audio_buffer: list[np.ndarray] = field(default_factory=list)  # 待处理的音频片段
    pending_audio_total_samples: int = 0  # 缓存音频总采样数

    # --------------------------------------------------------------------------
    # 纪要存储（仅采信 2-pass 结果）
    # --------------------------------------------------------------------------
    final_segments: list[TranscriptSegment] = field(default_factory=list)  # 最终归档纪要
    realtime_segments: list[StreamingSegment] = field(default_factory=list)  # 实时展示片段

    # --------------------------------------------------------------------------
    # 统计信息
    # --------------------------------------------------------------------------
    segment_id: int = 0  # 分句计数器
    twopass_count: int = 0  # 2-pass 执行次数
    vad_silence_trigger_count: int = 0  # VAD 静音触发次数
    long_speech_trigger_count: int = 0  # 长语音触发次数

    @property
    def speech_duration_ms(self) -> float:
        """计算当前连续语音段时长（毫秒）"""
        if self.speech_start_time is None:
            return 0
        return (time.time() - self.speech_start_time) * 1000

    @property
    def silence_duration_ms(self) -> float:
        """计算当前静音段时长（毫秒）"""
        if self.silence_start_time is None:
            return 0
        return (time.time() - self.silence_start_time) * 1000

    def reset_speech_state(self):
        """重置语音状态（进入静音时调用）"""
        self.speech_start_time = None
        self.total_speech_duration_ms = 0

    def reset_silence_state(self):
        """重置静音状态（进入语音时调用）"""
        self.silence_start_time = None


# ==============================================================================
# 引擎实现
# ==============================================================================

# 模块级单例
_instance: Optional["RealtimeMeetingEngine"] = None
_model_loaded: bool = False


def get_realtime_engine() -> "RealtimeMeetingEngine":
    """获取实时会议引擎单例"""
    global _instance
    if _instance is None:
        _instance = RealtimeMeetingEngine()
        _instance.load()
    return _instance


class RealtimeMeetingEngine(BaseAsrEngine):
    """
    实时会议纪要流式双路识别引擎

    采用 FunASR paraformer-zh-streaming 模型，结合 FSMN-VAD 实现：
    - 1-pass 流式识别：实时输出滚动字幕（用于展示）
    - 2-pass 全局修正：对完整句子进行全局推理（用于归档）

    特性：
    - 严格遵循 VAD 断句逻辑
    - 三种 2-pass 触发机制（静音超时、长语音强制分片、会话结束）
    - 结构化纪要输出，每条分句独立存储
    """

    engine_name = "realtime_meeting"

    # 类级别的模型实例（所有会话共享）
    _asr_model: Optional[Any] = None
    _vad_model: Optional[Any] = None
    _model_initialized: bool = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = detect_device()
        self._sessions: dict[str, RealtimeSessionState] = {}

    @property
    def asr_model(self) -> Any:
        """获取 ASR 模型"""
        return self._asr_model

    @property
    def vad_model(self) -> Any:
        """获取 VAD 模型"""
        return self._vad_model

    def load(self) -> None:
        """
        加载模型（类级别，只加载一次）

        加载两个独立模型：
        1. paraformer-zh-streaming: 流式 ASR 模型
        2. fsmn-vad: 静音检测模型
        """
        if RealtimeMeetingEngine._model_initialized:
            logger.debug("模型已加载，跳过")
            return

        try:
            from funasr import AutoModel
        except ImportError as exc:
            logger.error("FunASR 导入失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        model_cache_dir = str(self.settings.model_cache_dir)

        # 加载 ASR 流式模型
        try:
            asr_kwargs = {
                "model": MODEL_NAME,
                "device": self.device,
                "hub": "ms",
                "model_cache_dir": model_cache_dir,
                "disable_update": True,
            }
            RealtimeMeetingEngine._asr_model = AutoModel(**asr_kwargs)
            logger.info("ASR 模型加载完成: %s", MODEL_NAME)
        except Exception as exc:
            logger.error("ASR 模型加载失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        # 加载 VAD 模型
        try:
            vad_kwargs = {
                "model": VAD_MODEL_NAME,
                "device": self.device,
                "hub": "ms",
                "model_cache_dir": model_cache_dir,
                "disable_update": True,
            }
            RealtimeMeetingEngine._vad_model = AutoModel(**vad_kwargs)
            logger.info("VAD 模型加载完成: %s", VAD_MODEL_NAME)
        except Exception as exc:
            logger.error("VAD 模型加载失败: %s", exc)
            raise AppError(ERRORS["MODEL_LOAD_FAILED"]) from exc

        RealtimeMeetingEngine._model_initialized = True
        logger.info("实时会议引擎模型加载完成，device=%s", self.device)

    def create_session(
        self,
        chunk_size: Optional[list[int]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> str:
        """
        创建实时会议会话

        Args:
            chunk_size: chunk 大小配置，默认 [0, 10, 5]
            sample_rate: 采样率，默认 16000

        Returns:
            session_id: 会话唯一标识
        """
        # 确保模型已加载
        self.load()

        session_id = uuid4().hex

        # 验证并设置 chunk_size
        if chunk_size is None:
            chunk_size = CHUNK_SIZE.copy()

        # 确保 chunk_size 是有效列表
        if not isinstance(chunk_size, (list, tuple)) or len(chunk_size) < 3:
            logger.warning("无效 chunk_size: %s，使用默认值", chunk_size)
            chunk_size = CHUNK_SIZE.copy()

        # 初始化会话状态
        self._sessions[session_id] = RealtimeSessionState(
            session_id=session_id,
            chunk_size=chunk_size,
            sample_rate=sample_rate,
        )

        logger.info(
            "创建实时会议会话: session_id=%s, chunk_size=%s, sample_rate=%s",
            session_id, chunk_size, sample_rate
        )
        return session_id

    def close_session(self, session_id: str) -> None:
        """
        关闭会话并执行最后的 2-pass

        Args:
            session_id: 会话ID
        """
        if session_id not in self._sessions:
            logger.warning("会话不存在: %s", session_id)
            return

        session = self._sessions[session_id]

        # 执行会话结束的 2-pass 兜底
        if session.pending_audio_total_samples > 0:
            logger.info("会话结束，执行最后的 2-pass: session_id=%s", session_id)
            self._execute_twopass(session, TwoPassTrigger.SESSION_END)

        del self._sessions[session_id]
        logger.info("关闭会话: session_id=%s, 2-pass执行次数=%d", session_id, session.twopass_count)

    # ===========================================================================
    # 核心处理逻辑
    # ===========================================================================

    async def process_audio_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        is_final: bool = False,
    ) -> Optional[StreamingSegment]:
        """
        处理单个音频块（核心处理函数）

        处理流程：
        1. VAD 检测是否为语音/静音
        2. 更新会话状态（静音计时、语音计时）
        3. 执行 1-pass 流式识别（实时展示）
        4. 检查 2-pass 触发条件
        5. 执行 2-pass 全局修正（如需）

        Args:
            session_id: 会话ID
            audio_chunk: 音频数据（float32, 16kHz, 单声道）
            is_final: 是否为最后一个音频块

        Returns:
            StreamingSegment: 实时识别结果（如有新文本）
        """
        # 获取或创建会话
        if session_id not in self._sessions:
            session_id = self.create_session()

        session = self._sessions[session_id]

        # --------------------------------------------------------------------------
        # Step 1: VAD 检测
        # --------------------------------------------------------------------------
        is_speech = await self._detect_vad(session, audio_chunk)

        if is_speech:
            # 检测到语音
            await self._handle_speech_detected(session, audio_chunk)
        else:
            # 检测到静音
            await self._handle_silence_detected(session)

        # --------------------------------------------------------------------------
        # Step 2: 执行 1-pass 流式识别（实时展示）
        # --------------------------------------------------------------------------
        segment = await self._execute_1pass(session, audio_chunk)

        # --------------------------------------------------------------------------
        # Step 3: 检查 2-pass 触发条件
        # --------------------------------------------------------------------------
        trigger = self._check_twopass_trigger(session, is_final)

        if trigger is not None:
            # 有缓存音频，执行 2-pass
            if session.pending_audio_total_samples > 0:
                self._execute_twopass(session, trigger)

        # --------------------------------------------------------------------------
        # Step 4: 会话结束处理
        # --------------------------------------------------------------------------
        if is_final:
            # 执行最后的 2-pass 兜底
            if session.pending_audio_total_samples > 0:
                self._execute_twopass(session, TwoPassTrigger.SESSION_END)

        return segment

    async def _detect_vad(self, session: RealtimeSessionState, audio_chunk: np.ndarray) -> bool:
        """
        VAD 检测：判断当前音频是否为语音

        Args:
            session: 会话状态
            audio_chunk: 音频数据

        Returns:
            bool: True=语音, False=静音
        """
        try:
            # 使用 VAD 模型检测
            result = await asyncio.to_thread(
                self.vad_model.generate,
                input=audio_chunk,
                cache=session.vad_cache,
                is_final=False,
                chunk_size=session.chunk_size,
            )

            # 更新 VAD 缓存
            if result and isinstance(result, list) and len(result) > 0:
                session.vad_cache = result[0].get("cache", {})

            # FunASR VAD 返回格式检查
            # result 示例: [{"text": "...", "cache": {...}, "segment": {...}}]
            if result and isinstance(result, list):
                # 检查是否有语音段输出
                for item in result:
                    if isinstance(item, dict):
                        # 检测到语音标记
                        if item.get("text") or item.get("segment"):
                            return True

            # 简单检查：音频能量阈值作为备用
            audio_rms = np.sqrt(np.mean(audio_chunk ** 2))
            return audio_rms > 0.01  # RMS > 0.01 认为是语音

        except Exception as exc:
            logger.error("VAD 检测失败: %s", exc)
            # VAD 失败时保守返回有语音
            return True

    async def _handle_speech_detected(
        self,
        session: RealtimeSessionState,
        audio_chunk: np.ndarray,
    ) -> None:
        """
        处理语音检测事件

        - 切换状态为 SPEECH
        - 记录语音开始时间
        - 累加音频到缓存
        """
        # 状态转换
        if session.session_state == SessionState.SILENCE:
            # 从静音切换到语音，重置静音计时
            session.reset_silence_state()

        session.session_state = SessionState.SPEECH

        # 记录语音开始时间
        if session.speech_start_time is None:
            session.speech_start_time = time.time()

        # 累加语音时长
        chunk_duration_ms = len(audio_chunk) / session.sample_rate * 1000
        session.total_speech_duration_ms += chunk_duration_ms

        # 缓存音频（用于后续 2-pass）
        session.pending_audio_buffer.append(audio_chunk.copy())
        session.pending_audio_total_samples += len(audio_chunk)

    async def _handle_silence_detected(self, session: RealtimeSessionState) -> None:
        """
        处理静音检测事件

        - 切换状态为 SILENCE
        - 记录静音开始时间
        - 重置语音计时
        """
        # 状态转换
        if session.session_state == SessionState.SPEECH:
            # 从语音切换到静音，重置语音计时
            session.reset_speech_state()

        session.session_state = SessionState.SILENCE

        # 记录静音开始时间
        if session.silence_start_time is None:
            session.silence_start_time = time.time()

    async def _execute_1pass(
        self,
        session: RealtimeSessionState,
        audio_chunk: np.ndarray,
    ) -> Optional[StreamingSegment]:
        """
        执行 1-pass 流式识别（实时输出）

        使用 Paraformer 流式模型，实时输出滚动字幕。
        注意：1-pass 结果仅用于实时展示，不作为最终纪要。

        Args:
            session: 会话状态
            audio_chunk: 音频数据

        Returns:
            StreamingSegment: 识别片段（如有新文本）
        """
        try:
            # 调用 ASR 流式推理
            result = await asyncio.to_thread(
                self.asr_model.generate,
                input=audio_chunk,
                cache=session.asr_cache,
                is_final=False,
                chunk_size=session.chunk_size,
                encoder_chunk_look_back=4,  # 回看 4 个 chunk
                decoder_chunk_look_back=1,  # 回看 1 个 chunk
            )

            # 更新 ASR 缓存
            if result and isinstance(result, list) and len(result) > 0:
                session.asr_cache = result[0].get("cache", {})

            # 检查是否有文本输出
            if not result or not result[0].get("text"):
                return None

            text = result[0]["text"].strip()
            if not text:
                return None

            # 计算增量文本（用于展示）
            prev_text = session.last_1pass_text
            if prev_text and text.startswith(prev_text):
                new_text = text[len(prev_text):]
            else:
                new_text = text

            # 更新状态
            session.current_1pass_text = text
            session.last_1pass_text = text

            # 计算时间戳
            segment_id = session.segment_id
            session.segment_id += 1

            segment = StreamingSegment(
                segment_id=segment_id,
                text=new_text,
                full_text=text,
                start_ms=0,
                end_ms=0,
                is_final=False,
                speaker="spk0",
            )

            # 记录实时片段
            session.realtime_segments.append(segment)

            logger.debug(
                "1-pass 输出: session_id=%s, segment_id=%d, text='%s'",
                session.session_id, segment_id, new_text
            )

            return segment

        except Exception as exc:
            logger.error("1-pass 执行失败: session_id=%s, error=%s", session.session_id, exc)
            return None

    def _check_twopass_trigger(
        self,
        session: RealtimeSessionState,
        is_final: bool = False,
    ) -> Optional[TwoPassTrigger]:
        """
        检查 2-pass 触发条件

        三种触发机制：
        1. VAD 静音超时（默认 0.5s）- 自然断句
        2. 单人连续说话最长 6s - 防止缓存漂移
        3. 会话结束 - 兜底处理

        Args:
            session: 会话状态
            is_final: 是否为最后一个音频块

        Returns:
            TwoPassTrigger: 触发原因，None=不触发
        """
        # 触发机制 1: VAD 静音超时触发 2-pass
        # ----------------------------------------------------------
        # 业务意义：用户在说话过程中出现停顿（换气、思考），
        # 超过 0.5s 静音后，认为是一个完整的句子结束。
        # 此时执行 2-pass 对该句子进行全局修正，输出更准确的结果。
        if session.session_state == SessionState.SILENCE:
            silence_duration = session.silence_duration_ms
            if silence_duration >= VAD_SILENCE_TIMEOUT_MS:
                logger.info(
                    "2-pass 触发 [VAD静音]: session_id=%s, 静音时长=%.1fms",
                    session.session_id, silence_duration
                )
                session.vad_silence_trigger_count += 1
                return TwoPassTrigger.VAD_SILENCE

        # 触发机制 2: 长语音强制分片触发 2-pass
        # ----------------------------------------------------------
        # 业务意义：单人连续说话超过 6s，ASR 模型的注意力机制
        # 可能出现"漂移"，导致识别错误。
        # 此时强制分片，对已有内容执行 2-pass，然后继续识别后续内容。
        # 防止长语音导致的识别质量下降。
        speech_duration = session.speech_duration_ms
        if speech_duration >= MAX_SPEECH_DURATION_MS:
            logger.info(
                "2-pass 触发 [长语音分片]: session_id=%s, 语音时长=%.1fms",
                session.session_id, speech_duration
            )
            session.long_speech_trigger_count += 1
            return TwoPassTrigger.LONG_SPEECH

        # 触发机制 3: 会话结束兜底触发 2-pass
        # ----------------------------------------------------------
        # 业务意义：会议/录音结束时，可能还有未处理的音频缓存。
        # 此时必须执行 2-pass，确保所有内容都被识别和存档。
        if is_final and session.pending_audio_total_samples > 0:
            logger.info(
                "2-pass 触发 [会话结束]: session_id=%s, 剩余缓存=%.1fs",
                session.session_id,
                session.pending_audio_total_samples / session.sample_rate
            )
            return TwoPassTrigger.SESSION_END

        return None

    def _execute_twopass(
        self,
        session: RealtimeSessionState,
        trigger: TwoPassTrigger,
    ) -> None:
        """
        执行 2-pass 全局修正

        关键规范：
        - input=None：使用缓存的音频数据
        - is_final=True：执行完整的全局推理
        - 执行后重置 ASR cache 和缓存状态

        2-pass 结果作为最终会议纪要存档。

        Args:
            session: 会话状态
            trigger: 触发原因
        """
        if session.pending_audio_total_samples == 0:
            logger.debug("无待处理音频，跳过 2-pass: session_id=%s", session.session_id)
            return

        try:
            # 合并缓存的音频
            audio_data = np.concatenate(session.pending_audio_buffer)
            logger.info(
                "执行 2-pass: session_id=%s, trigger=%s, 音频时长=%.2fs",
                session.session_id, trigger.value, len(audio_data) / session.sample_rate
            )

            # 执行 2-pass 推理
            # ----------------------------------------------------------
            # 关键参数说明：
            # - input: 音频数据（合并后的完整句子）
            # - cache: None（不使用缓存，强制全局推理）
            # - is_final=True: 执行完整推理
            # - chunk_size: [0, 0, 0] 表示非流式模式
            # ----------------------------------------------------------
            result = self.asr_model.generate(
                input=audio_data,
                cache=None,  # 关键：input=None 时使用缓存，但这里我们传音频数据
                is_final=True,  # 关键：执行全局修正
                chunk_size=[0, 0, 0],  # 关键：非流式模式
            )

            # 处理结果
            if result and isinstance(result, list) and len(result) > 0:
                text = result[0].get("text", "").strip()

                if text:
                    # 创建最终纪要分句
                    segment = TranscriptSegment(
                        speaker="spk0",
                        start_ms=int(time.time() * 1000),  # 简化时间戳
                        end_ms=int(time.time() * 1000),
                        text=text,
                    )

                    # 记录到最终纪要（仅采信 2-pass 结果）
                    session.final_segments.append(segment)

                    logger.info(
                        "2-pass 完成: session_id=%s, segment_id=%d, text='%s'",
                        session.session_id, len(session.final_segments), text
                    )

            # 执行后重置状态
            # ----------------------------------------------------------
            # 重置 ASR cache：清除流式推理的内部状态，
            # 避免对下一句的识别产生影响。
            # ----------------------------------------------------------
            session.asr_cache = {}

            # 清空音频缓存
            session.pending_audio_buffer = []
            session.pending_audio_total_samples = 0

            # 重置语音状态
            session.reset_speech_state()

            # 更新统计
            session.twopass_count += 1

        except Exception as exc:
            logger.error(
                "2-pass 执行失败: session_id=%s, trigger=%s, error=%s",
                session.session_id, trigger.value, exc
            )
            # 失败时也重置状态，避免死锁
            session.asr_cache = {}
            session.pending_audio_buffer = []
            session.pending_audio_total_samples = 0

    # ===========================================================================
    # 批量处理接口
    # ===========================================================================

    async def process_audio_file(
        self,
        audio_path: Path,
        session_id: Optional[str] = None,
    ) -> EngineResult:
        """
        处理音频文件（模拟实时流输入）

        将音频文件分块模拟实时输入，用于测试和离线处理。

        Args:
            audio_path: 音频文件路径
            session_id: 可选的会话ID

        Returns:
            EngineResult: 最终识别结果（包含 2-pass 结果）
        """
        import soundfile as sf

        # 加载音频
        audio_data, sr = sf.read(audio_path, dtype='float32')

        # 立体声转单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # 重采样到 16kHz
        if sr != DEFAULT_SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)

        # 创建会话
        if session_id is None:
            session_id = self.create_session()

        # 分块处理（模拟实时流）
        chunk_samples = SAMPLES_PER_CHUNK
        all_realtime_segments = []

        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            is_final = (i + chunk_samples >= len(audio_data))

            segment = await self.process_audio_chunk(session_id, chunk, is_final)
            if segment:
                all_realtime_segments.append(segment)

        # 关闭会话（触发最后的 2-pass）
        self.close_session(session_id)

        # 获取最终结果
        session = self._sessions.get(session_id)
        if session:
            final_text = " ".join(s.text for s in session.final_segments if s.text)
        else:
            final_text = ""

        return EngineResult(
            text=final_text,
            segments=session.final_segments if session else [],
            language="zh",
            metadata={
                "session_id": session_id,
                "twopass_count": session.twopass_count if session else 0,
                "vad_silence_triggers": session.vad_silence_trigger_count if session else 0,
                "long_speech_triggers": session.long_speech_trigger_count if session else 0,
            },
        )

    # ===========================================================================
    # 工具方法
    # ===========================================================================

    def get_session_state(self, session_id: str) -> Optional[RealtimeSessionState]:
        """获取会话状态"""
        return self._sessions.get(session_id)

    def get_final_transcript(self, session_id: str) -> Optional[str]:
        """
        获取最终纪要文本（仅 2-pass 结果）

        Returns:
            str: 合并后的最终纪要文本
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        return " ".join(s.text for s in session.final_segments if s.text)

    def get_realtime_text(self, session_id: str) -> Optional[str]:
        """
        获取实时文本（1-pass 结果）

        注意：1-pass 结果仅用于实时展示，不作为最终纪要。

        Returns:
            str: 当前实时识别文本
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        return session.current_1pass_text

    def export_meeting_notes(self, session_id: str) -> Optional[dict]:
        """
        导出结构化会议纪要

        Returns:
            dict: 结构化纪要数据
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        return {
            "session_id": session_id,
            "created_at": session.created_at,
            "total_segments": len(session.final_segments),
            "twopass_count": session.twopass_count,
            "vad_silence_trigger_count": session.vad_silence_trigger_count,
            "long_speech_trigger_count": session.long_speech_trigger_count,
            "segments": [
                {
                    "segment_id": i,
                    "speaker": seg.speaker,
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "text": seg.text,
                }
                for i, seg in enumerate(session.final_segments)
            ],
            "full_text": " ".join(s.text for s in session.final_segments if s.text),
        }
