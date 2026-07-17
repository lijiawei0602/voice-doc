from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.services.asr.funasr_engine import get_funasr_engine, DEFAULT_CHUNK_SIZE

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])


class StreamingConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket 连接建立: session_id=%s", session_id)

    async def disconnect(self, session_id: str) -> None:
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info("WebSocket 连接断开: session_id=%s", session_id)

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


manager = StreamingConnectionManager()


@router.websocket("/ws")
async def websocket_stream_transcribe(websocket: WebSocket):
    """WebSocket 流式语音识别接口

    客户端连接后，需要发送以下 JSON 消息来创建会话：
    {
        "type": "start",
        "chunk_size": [0, 10, 5],  // 可选，默认 [0, 10, 5]
        "sample_rate": 16000        // 可选，默认 16000
    }

    发送音频数据（二进制 PCM 数据，16kHz, 16位）：
    - 直接发送二进制音频数据

    发送结束信号：
    {"type": "end"}

    服务端返回识别结果（增量输出）：
    {
        "type": "segment",
        "segment_id": 0,
        "text": "新增的文字",
        "is_final": false
    }

    服务端返回最终结果：
    {
        "type": "finished",
        "session_id": "xxx",
        "full_text": "完整识别文字"
    }
    """
    import numpy as np

    session_id: Optional[str] = None
    engine = get_funasr_engine()

    try:
        await websocket.accept()
        logger.info("WebSocket 连接已接受")

        while True:
            try:
                data = await websocket.receive()

                if "text" in data:
                    message = json.loads(data["text"])
                    msg_type = message.get("type", "")

                    if msg_type == "start":
                        chunk_size = message.get("chunk_size", DEFAULT_CHUNK_SIZE)
                        session_id = engine.create_streaming_session(chunk_size=chunk_size)
                        await manager.send_json(
                            session_id,
                            {"type": "started", "session_id": session_id}
                        )
                        logger.info("流式识别会话创建: session_id=%s, chunk_size=%s", session_id, chunk_size)

                    elif msg_type == "end":
                        if session_id:
                            await manager.send_json(
                                session_id,
                                {"type": "finished", "session_id": session_id}
                            )

                    elif msg_type == "close":
                        close_session_id = message.get("session_id", session_id)
                        if close_session_id:
                            engine.close_streaming_session(close_session_id)
                            await manager.disconnect(close_session_id)
                        break

                elif "bytes" in data:
                    audio_bytes = data["bytes"]
                    if session_id and audio_bytes:
                        # 接收 PCM 音频数据并转换为 numpy 数组
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0

                        # 流式识别（这里模拟增量输出）
                        segment = await engine.stream_transcribe(
                            session_id, audio_float, is_final=False
                        )

                        if segment and segment.text:
                            await manager.send_json(
                                session_id,
                                {
                                    "type": "segment",
                                    "segment_id": segment.segment_id,
                                    "text": segment.text,
                                    "is_final": segment.is_final,
                                }
                            )

            except WebSocketDisconnect:
                logger.info("WebSocket 客户端断开连接: session_id=%s", session_id)
                break
            except Exception as exc:
                logger.error("WebSocket 处理异常: %s", exc)
                if session_id:
                    await manager.send_json(
                        session_id,
                        {"type": "error", "error": str(exc)}
                    )
                break

    except Exception as exc:
        logger.error("WebSocket 连接异常: %s", exc)
    finally:
        if session_id:
            engine.close_streaming_session(session_id)
            await manager.disconnect(session_id)
        logger.info("WebSocket 会话结束: session_id=%s", session_id)
