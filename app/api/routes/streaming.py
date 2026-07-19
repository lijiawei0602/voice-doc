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
    connected = True

    try:
        await websocket.accept()
        logger.info("WebSocket 连接已接受")

        while connected:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("WebSocket 客户端断开连接: session_id=%s", session_id)
                break

            try:
                if "text" in data:
                    message = json.loads(data["text"])
                    msg_type = message.get("type", "")

                    if msg_type == "start":
                        chunk_size = message.get("chunk_size")
                        # 解析字符串形式的 chunk_size
                        if chunk_size is not None:
                            if isinstance(chunk_size, str):
                                try:
                                    chunk_size = json.loads(chunk_size)
                                except json.JSONDecodeError:
                                    chunk_size = DEFAULT_CHUNK_SIZE
                            elif not isinstance(chunk_size, (list, tuple)):
                                chunk_size = DEFAULT_CHUNK_SIZE
                        else:
                            chunk_size = DEFAULT_CHUNK_SIZE
                        
                        logger.info("收到 start 消息: %s", message)
                        session_id = engine.create_streaming_session(chunk_size=chunk_size)
                        await websocket.send_json({
                            "type": "started",
                            "session_id": session_id
                        })
                        logger.info("流式识别会话创建成功: session_id=%s, chunk_size=%s", session_id, chunk_size)

                    elif msg_type == "end":
                        if session_id:
                            await websocket.send_json({
                                "type": "finished",
                                "session_id": session_id
                            })

                    elif msg_type == "close":
                        close_session_id = message.get("session_id", session_id)
                        if close_session_id:
                            engine.close_streaming_session(close_session_id)
                        connected = False
                        break

                elif "bytes" in data:
                    audio_bytes = data["bytes"]
                    if session_id and audio_bytes:
                        logger.info("收到音频数据: session_id=%s, bytes_length=%s", 
                                   session_id, len(audio_bytes))
                        
                        # 接收 PCM 音频数据并转换为 numpy 数组
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0
                        
                        # 检查音频数据质量
                        audio_sum = float(np.abs(audio_float).sum())
                        audio_avg = audio_sum / len(audio_float) if len(audio_float) > 0 else 0
                        logger.info("音频数据质量: session_id=%s, avg=%.6f, min=%.6f, max=%.6f",
                                   session_id, audio_avg, float(audio_float.min()), float(audio_float.max()))

                        # 流式识别
                        segment = await engine.stream_transcribe(
                            session_id, audio_float, is_final=False
                        )

                        if segment and segment.text:
                            logger.info("发送识别结果: session_id=%s, segment_id=%s, text='%s', full_text='%s'",
                                       session_id, segment.segment_id, segment.text, segment.full_text)
                            await websocket.send_json({
                                "type": "segment",
                                "segment_id": segment.segment_id,
                                "text": segment.text,
                                "full_text": segment.full_text,
                                "is_final": segment.is_final,
                            })
                        else:
                            logger.info("无识别结果: session_id=%s, segment=%s", 
                                       session_id, segment)

            except WebSocketDisconnect:
                logger.info("WebSocket 客户端断开连接: session_id=%s", session_id)
                break
            except Exception as exc:
                logger.error("WebSocket 处理异常: %s", exc)
                try:
                    if session_id:
                        await websocket.send_json({
                            "type": "error",
                            "error": str(exc)
                        })
                except Exception:
                    pass
                break

    except Exception as exc:
        logger.error("WebSocket 连接异常: %s", exc)
    finally:
        if session_id:
            engine.close_streaming_session(session_id)
        logger.info("WebSocket 会话结束: session_id=%s", session_id)


@router.post("/test")
async def test_streaming():
    """测试流式识别接口
    
    使用项目中的示例音频文件测试流式识别是否正常工作。
    支持格式: wav, m4a, mp3, webm, flac, ogg 等
    """
    import soundfile as sf
    from pathlib import Path
    
    engine = get_funasr_engine()
    engine.load_streaming_model()
    
    # 查找示例音频文件
    sample_files = [
        Path("models/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"),
        Path("./data/test_audio.wav"),
        Path("./data/test_audio.m4a"),
        Path("./data/test_audio.webm"),
        Path("./data/test_audio.mp3"),
        Path("./test_audio.wav"),
        Path("./test_audio.m4a"),
        Path("./test_audio.webm"),
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
            "suggestion": "请上传音频文件到 ./data/ 目录，支持 wav/m4a/webm/mp3/flac 等格式"
        }
    
    logger.info("开始测试流式识别，使用文件: %s", sample_file)
    
    # 读取音频文件（支持多种格式）
    try:
        import librosa
        
        # soundfile 支持的格式
        soundfile_exts = {'.wav', '.flac', '.ogg'}
        
        if sample_file.suffix.lower() in soundfile_exts:
            # 使用 soundfile 读取
            audio_data, sr = sf.read(sample_file, dtype='float32')
            logger.info("使用 soundfile 读取: %s", sample_file)
            logger.info("音频信息: sample_rate=%s, channels=%s, length=%s", sr, 
                       audio_data.shape[1] if len(audio_data.shape) > 1 else 1, len(audio_data))
            
            # 立体声转单声道
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 重采样到 16kHz
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                logger.info("重采样后: sample_rate=16000, length=%s", len(audio_data))
        else:
            # 使用 librosa 读取（支持 m4a, mp3, webm 等）
            audio_data, sr = librosa.load(sample_file, sr=16000, mono=True)
            logger.info("使用 librosa 读取: %s", sample_file)
            logger.info("音频信息: sample_rate=%s, length=%s", sr, len(audio_data))
            
    except Exception as e:
        logger.error("读取音频文件失败: %s", e)
        return {"status": "error", "message": f"读取音频文件失败: {e}"}
    
    # 创建流式会话并测试
    session_id = engine.create_streaming_session()
    logger.info("创建测试会话: %s", session_id)
    
    # 分块处理
    chunk_size = 16000  # 1秒音频
    all_segments = []
    full_text = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        is_final = (i + chunk_size >= len(audio_data))
        
        segment = await engine.stream_transcribe(
            session_id, chunk, is_final=is_final
        )
        
        if segment and segment.text:
            all_segments.append(segment)
            full_text.append(segment.text)
            logger.info("识别片段 %d: '%s'", segment.segment_id, segment.text)
    
    engine.close_streaming_session(session_id)
    
    result_text = "".join(full_text)
    logger.info("测试完成，识别结果: %s", result_text)
    
    return {
        "status": "success",
        "session_id": session_id,
        "audio_file": str(sample_file),
        "sample_rate": 16000,
        "audio_length": len(audio_data),
        "segment_count": len(all_segments),
        "full_text": result_text,
        "segments": [
            {
                "id": s.segment_id,
                "text": s.text,
                "is_final": s.is_final
            }
            for s in all_segments
        ]
    }
