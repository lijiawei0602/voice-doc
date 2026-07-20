"""
实时会议纪要 API 路由
====================

提供 WebSocket 接口，支持实时会议纪要的流式双路识别：
- 1-pass: 实时滚动字幕展示
- 2-pass: 全局修正结果作为最终纪要存档

Author: Voice-Doc Team
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.asr.realtime_meeting_engine import (
    get_realtime_engine,
    CHUNK_SIZE,
    TwoPassTrigger,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/realtime-meeting", tags=["realtime-meeting"])


# ==============================================================================
# 请求/响应模型
# ==============================================================================

class MeetingStartRequest(BaseModel):
    """会议开始请求"""
    session_id: Optional[str] = None
    chunk_size: Optional[list[int]] = None
    sample_rate: int = 16000


class MeetingEndRequest(BaseModel):
    """会议结束请求"""
    session_id: str


class TwopassTriggerInfo(BaseModel):
    """2-pass 触发信息"""
    trigger: str
    message: str


# ==============================================================================
# WebSocket 连接管理器
# ==============================================================================

class MeetingConnectionManager:
    """会议 WebSocket 连接管理器"""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("会议 WebSocket 连接建立: session_id=%s", session_id)

    async def disconnect(self, session_id: str) -> None:
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info("会议 WebSocket 连接断开: session_id=%s", session_id)

    async def send_json(self, session_id: str, data: dict) -> bool:
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(data)
                return True
            except Exception as exc:
                logger.error("发送消息失败: session_id=%s, error=%s", session_id, exc)
                return False
        return False


manager = MeetingConnectionManager()


# ==============================================================================
# WebSocket 接口
# ==============================================================================

@router.websocket("/ws")
async def websocket_realtime_meeting(websocket: WebSocket):
    """
    实时会议纪要 WebSocket 接口

    协议说明：
    1. 客户端发送 {"type": "start", ...} 创建会议会话
    2. 客户端发送二进制 PCM 音频数据（16kHz, 16位, 单声道）
    3. 服务端返回实时识别结果（1-pass）
    4. 服务端返回 2-pass 触发通知和最终结果
    5. 客户端发送 {"type": "end"} 结束会议

    客户端消息格式：
    - start: {"type": "start", "chunk_size": [0, 10, 5], "sample_rate": 16000}
    - audio: 二进制 PCM 数据
    - end: {"type": "end"}

    服务端消息格式：
    - started: {"type": "started", "session_id": "xxx"}
    - realtime: {"type": "realtime", "segment_id": 0, "text": "新增文字", "full_text": "完整文字"}
    - twopass_trigger: {"type": "twopass_trigger", "trigger": "vad_silence", "message": "..."}
    - final_segment: {"type": "final_segment", "segment_id": 0, "speaker": "spk0", "text": "最终文字"}
    - finished: {"type": "finished", "session_id": "xxx", "full_text": "完整纪要", "segments": [...]}
    - error: {"type": "error", "error": "错误信息"}
    """
    import numpy as np

    session_id: Optional[str] = None
    engine = get_realtime_engine()
    connected = True

    try:
        await websocket.accept()
        logger.info("会议 WebSocket 连接已接受")

        while connected:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("会议 WebSocket 客户端断开: session_id=%s", session_id)
                break

            try:
                # 处理文本消息
                if "text" in data:
                    message = json.loads(data["text"])
                    msg_type = message.get("type", "")

                    if msg_type == "start":
                        # 创建会议会话
                        chunk_size = message.get("chunk_size", CHUNK_SIZE)
                        sample_rate = message.get("sample_rate", 16000)

                        logger.info("收到会议开始请求: %s", message)

                        session_id = engine.create_session(
                            chunk_size=chunk_size,
                            sample_rate=sample_rate,
                        )

                        await websocket.send_json({
                            "type": "started",
                            "session_id": session_id,
                            "chunk_size": chunk_size,
                            "sample_rate": sample_rate,
                        })

                        logger.info("会议会话创建成功: session_id=%s", session_id)

                    elif msg_type == "end":
                        # 结束会议
                        if session_id:
                            # 发送最终的 2-pass 结果
                            final_text = engine.get_final_transcript(session_id)
                            meeting_notes = engine.export_meeting_notes(session_id)

                            await websocket.send_json({
                                "type": "finished",
                                "session_id": session_id,
                                "full_text": final_text or "",
                                "segments": meeting_notes.get("segments", []) if meeting_notes else [],
                                "stats": {
                                    "twopass_count": meeting_notes.get("twopass_count", 0) if meeting_notes else 0,
                                    "vad_silence_triggers": meeting_notes.get("vad_silence_trigger_count", 0) if meeting_notes else 0,
                                    "long_speech_triggers": meeting_notes.get("long_speech_trigger_count", 0) if meeting_notes else 0,
                                } if meeting_notes else {},
                            })

                            # 关闭会话
                            engine.close_session(session_id)
                            logger.info("会议结束: session_id=%s", session_id)

                        connected = False
                        break

                    elif msg_type == "close":
                        # 强制关闭
                        close_session_id = message.get("session_id", session_id)
                        if close_session_id:
                            engine.close_session(close_session_id)
                        connected = False
                        break

                # 处理二进制音频数据
                elif "bytes" in data:
                    audio_bytes = data["bytes"]

                    if not session_id:
                        # 自动创建会话
                        session_id = engine.create_session()

                    if audio_bytes:
                        # 转换 PCM 数据
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0

                        logger.debug(
                            "收到音频数据: session_id=%s, samples=%d",
                            session_id, len(audio_float)
                        )

                        # 处理音频块
                        segment = await engine.process_audio_chunk(
                            session_id,
                            audio_float,
                            is_final=False,
                        )

                        # 发送 1-pass 实时结果
                        if segment and segment.text:
                            await websocket.send_json({
                                "type": "realtime",
                                "segment_id": segment.segment_id,
                                "text": segment.text,
                                "full_text": segment.full_text,
                            })

                        # 定期发送状态信息
                        session_state = engine.get_session_state(session_id)
                        if session_state:
                            # 发送 2-pass 触发通知（如有）
                            if session_state.twopass_count > 0:
                                last_trigger = None
                                if session_state.vad_silence_trigger_count > 0:
                                    last_trigger = TwoPassTrigger.VAD_SILENCE.value
                                elif session_state.long_speech_trigger_count > 0:
                                    last_trigger = TwoPassTrigger.LONG_SPEECH.value

                                if last_trigger:
                                    await websocket.send_json({
                                        "type": "twopass_trigger",
                                        "trigger": last_trigger,
                                        "message": f"2-pass 执行次数: {session_state.twopass_count}",
                                        "stats": {
                                            "vad_silence": session_state.vad_silence_trigger_count,
                                            "long_speech": session_state.long_speech_trigger_count,
                                        },
                                    })

                            # 发送最新的 2-pass 结果
                            if session_state.final_segments:
                                latest = session_state.final_segments[-1]
                                await websocket.send_json({
                                    "type": "final_segment",
                                    "segment_id": len(session_state.final_segments) - 1,
                                    "speaker": latest.speaker,
                                    "text": latest.text,
                                })

            except WebSocketDisconnect:
                logger.info("会议 WebSocket 断开: session_id=%s", session_id)
                break
            except Exception as exc:
                logger.error("处理异常: %s", exc)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(exc),
                    })
                except Exception:
                    pass
                break

    except Exception as exc:
        logger.error("会议 WebSocket 连接异常: %s", exc)
    finally:
        if session_id:
            # 确保关闭会话
            try:
                engine.close_session(session_id)
            except Exception:
                pass
        logger.info("会议 WebSocket 会话结束: session_id=%s", session_id)


# ==============================================================================
# REST 接口
# ==============================================================================

@router.post("/test")
async def test_realtime_meeting():
    """
    测试实时会议纪要接口

    使用项目中的示例音频文件测试双路识别是否正常工作。
    """
    from pathlib import Path
    import soundfile as sf
    import librosa

    engine = get_realtime_engine()

    # 查找示例音频文件
    sample_files = [
        Path("models/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"),
        Path("./data/test_audio.wav"),
        Path("./data/test_audio.m4a"),
        Path("./data/test_audio.mp3"),
    ]

    sample_file = None
    for path in sample_files:
        if path.exists():
            sample_file = path
            break

    if sample_file is None:
        return {
            "status": "error",
            "message": "未找到测试音频文件",
            "suggestion": "请上传音频文件到 ./data/ 目录",
        }

    logger.info("测试实时会议纪要，使用文件: %s", sample_file)

    # 读取音频
    try:
        if sample_file.suffix.lower() in {'.wav', '.flac', '.ogg'}:
            audio_data, sr = sf.read(sample_file, dtype='float32')
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
        else:
            audio_data, sr = librosa.load(sample_file, sr=16000, mono=True)

        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    except Exception as e:
        logger.error("读取音频失败: %s", e)
        return {"status": "error", "message": f"读取音频失败: {e}"}

    # 创建会话并测试
    session_id = engine.create_session()
    logger.info("创建测试会话: %s", session_id)

    # 分块处理
    chunk_samples = 16000  # 1秒
    all_realtime = []
    all_final = []

    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        is_final = (i + chunk_samples >= len(audio_data))

        segment = await engine.process_audio_chunk(session_id, chunk, is_final)

        if segment and segment.text:
            all_realtime.append({
                "segment_id": segment.segment_id,
                "text": segment.text,
            })

        # 获取 2-pass 结果
        session = engine.get_session_state(session_id)
        if session and session.final_segments:
            if len(session.final_segments) > len(all_final):
                latest = session.final_segments[-1]
                all_final.append({
                    "segment_id": len(all_final),
                    "speaker": latest.speaker,
                    "text": latest.text,
                })

    # 关闭会话
    engine.close_session(session_id)

    # 导出最终纪要
    meeting_notes = engine.export_meeting_notes(session_id)

    return {
        "status": "success",
        "session_id": session_id,
        "audio_file": str(sample_file),
        "audio_length": len(audio_data) / 16000,
        "realtime_segments": all_realtime,
        "final_segments": all_final,
        "meeting_notes": meeting_notes,
    }


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    engine = get_realtime_engine()
    session = engine.get_session_state(session_id)

    if session is None:
        return {"status": "error", "message": "会话不存在"}

    return {
        "status": "success",
        "session_id": session_id,
        "session_state": session.session_state.value,
        "speech_duration_ms": session.speech_duration_ms,
        "silence_duration_ms": session.silence_duration_ms,
        "pending_audio_samples": session.pending_audio_total_samples,
        "realtime_segments_count": len(session.realtime_segments),
        "final_segments_count": len(session.final_segments),
        "twopass_count": session.twopass_count,
        "vad_silence_triggers": session.vad_silence_trigger_count,
        "long_speech_triggers": session.long_speech_trigger_count,
        "current_1pass_text": session.current_1pass_text,
    }


@router.get("/sessions/{session_id}/transcript")
async def get_final_transcript(session_id: str):
    """获取最终纪要文本"""
    engine = get_realtime_engine()
    final_text = engine.get_final_transcript(session_id)
    meeting_notes = engine.export_meeting_notes(session_id)

    if final_text is None:
        return {"status": "error", "message": "会话不存在"}

    return {
        "status": "success",
        "session_id": session_id,
        "full_text": final_text,
        "segments": meeting_notes.get("segments", []) if meeting_notes else [],
        "stats": {
            "total_segments": meeting_notes.get("total_segments", 0) if meeting_notes else 0,
            "twopass_count": meeting_notes.get("twopass_count", 0) if meeting_notes else 0,
        },
    }
