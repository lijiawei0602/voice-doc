# Voice Doc Service

基于 Python + FastAPI 的离线语音转文字后端服务，提供两套本地推理方案：

- 主方案：`FunASR`，内置 ASR + VAD + 标点 + 说话人分离
- 备选方案：`Whisper + pyannote.audio`，适合多语言场景

## 1. 项目目录结构

```text
voice-doc/
├── app/
│   ├── api/
│   │   ├── router.py
│   │   └── routes/
│   │       ├── health.py
│   │       ├── tasks.py
│   │       └── transcriptions.py
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── logging.py
│   │   └── response.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── asr/
│   │   │   ├── base.py
│   │   │   ├── funasr_engine.py
│   │   │   └── whisper_engine.py
│   │   ├── audio/
│   │   │   ├── chunking.py
│   │   │   └── preprocess.py
│   │   ├── diarization/
│   │   │   └── pyannote_diarizer.py
│   │   ├── pipeline/
│   │   │   └── transcription_service.py
│   │   └── tasks/
│   │       └── task_manager.py
│   ├── utils/
│   │   ├── device.py
│   │   ├── files.py
│   │   └── json_store.py
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── results/
│   └── tasks/
├── logs/
├── scripts/
│   └── start.sh
├── .env.example
├── requirements.txt
├── requirements-funasr.txt
├── requirements-whisper.txt
└── README.md
```

## 2. 功能说明

- `FastAPI` RESTful 接口，自带 Swagger 文档：`/docs`
- 支持 `mp3/wav/flac/m4a/webm/ogg/aac`
- 自动预处理：重采样为 `16k/mono wav`，并做基础滤波与降噪
- 输出完整识别结果：
  - 全量文本
  - 分句时间戳
  - 每句对应说话人 `spk0/spk1/spk2`
  - 按说话人聚合后的完整台词
- 支持本地路径批量识别
- 支持长音频后台异步任务
- 支持任务状态查询
- 结果自动保存到 `data/results/*.json`
- 异常统一封装，标准化返回错误码

## 3. 安装依赖

### 3.1 FunASR 主方案

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-funasr.txt
```

### 3.2 Whisper 备选方案

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-whisper.txt
```

### 3.3 系统依赖

必须安装 `ffmpeg` 与 `ffprobe`：

```bash
brew install ffmpeg
```

## 4. 环境变量

```bash
cp .env.example .env
```

关键配置：

- `ENGINE=funasr` 或 `ENGINE=whisper`
- `ENABLE_GPU=true`
- `FORCE_CPU=false`
- `MODEL_CACHE_DIR=./models`
- `MAX_AUDIO_DURATION_SECONDS=14400`
- `AUDIO_CHUNK_SECONDS=1800`
- `FUNASR_MODEL=paraformer-zh`
- `FUNASR_SPK_MODEL=cam++`
- `WHISPER_MODEL=small`
- `PYANNOTE_MODEL=pyannote/speaker-diarization-3.1`

## 5. 启动服务

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

或直接运行：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
rm -rf .venv
uv venv --python=python3.10
source .venv/bin/activate
deactivate
uv pip install -r requirements.txt

uv run python scripts/preload_models.py
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
PRELOAD_MODELS_ON_STARTUP=false uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop
```

Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

## 6. 接口说明

### 6.1 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 6.2 上传音频同步识别

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/transcriptions/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/demo.wav"
```

### 6.3 本地路径批量识别

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/transcriptions/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "paths": [
      "/absolute/path/a.wav",
      "/absolute/path/b.mp3"
    ]
  }'
```

### 6.4 上传音频异步识别

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/tasks/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/meeting.wav"
```

### 6.5 查询任务状态

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/<task_id>"
```

## 7. 返回结果示例

```json
{
  "code": "SUCCESS",
  "message": "success",
  "data": {
    "task_id": "79c0f9a1c5f649c8bce39d1a7935e862",
    "engine": "funasr",
    "device": "cuda:0",
    "source": "meeting.wav",
    "duration_seconds": 31.218,
    "text": "你好，今天我们讨论一下项目排期。好的，我先汇报后端接口。",
    "segments": [
      {
        "speaker": "spk0",
        "start_ms": 0,
        "end_ms": 1850,
        "text": "你好，今天我们讨论一下项目排期。",
        "chunk_index": 0
      },
      {
        "speaker": "spk1",
        "start_ms": 1850,
        "end_ms": 3760,
        "text": "好的，我先汇报后端接口。",
        "chunk_index": 0
      }
    ],
    "speakers": [
      {
        "speaker": "spk0",
        "text": "你好，今天我们讨论一下项目排期。",
        "segments": []
      },
      {
        "speaker": "spk1",
        "text": "好的，我先汇报后端接口。",
        "segments": []
      }
    ],
    "result_path": "data/results/79c0f9a1c5f649c8bce39d1a7935e862.json",
    "metadata": {
      "chunk_count": 1
    }
  }
}
```

## 8. 两套方案切换

修改 `.env`：

```bash
ENGINE=funasr
```

或：

```bash
ENGINE=whisper
```

然后重启服务即可。

## 9. FunASR vs Whisper + pyannote

### FunASR 优点

- 一体化接口，ASR、VAD、标点、说话人分离在同一流水线
- 中文会议、访谈场景更轻量
- CPU 下通常也能运行
- 更适合作为默认私有部署方案

### FunASR 局限

- 不同模型在多语言泛化上不如 Whisper 灵活
- 超长音频分片时，跨片段说话人一致性可能略受影响

### Whisper + pyannote 优点

- Whisper 多语言覆盖更强
- pyannote 在多说话人切分方面更成熟
- 适合国际化或复杂音频场景

### Whisper + pyannote 局限

- 依赖更重
- GPU 需求更高
- 首次准备 pyannote 模型通常更复杂

## 10. GPU 部署说明

- 先安装匹配 CUDA 的 `torch` / `torchaudio`
- `ENABLE_GPU=true` 且机器检测到 CUDA 时自动走 GPU
- 未检测到 GPU 时自动降级到 CPU
- 如需强制 CPU，设置 `FORCE_CPU=true`

## 11. 模型下载说明

- 模型默认缓存到 `MODEL_CACHE_DIR`
- FunASR 首次运行会自动缓存模型
- Whisper 首次运行会缓存到 `WHISPER_DOWNLOAD_ROOT`
- pyannote 首次使用前建议先完成模型下载并缓存到本地
- 离线部署时，建议在联网机器预热模型缓存后整体拷贝到服务器

## 12. 错误码

统一返回：

- `SUCCESS`
- `INVALID_AUDIO_FORMAT`
- `FILE_TOO_LARGE`
- `AUDIO_TOO_LONG`
- `AUDIO_CORRUPTED`
- `MODEL_LOAD_FAILED`
- `TRANSCRIPTION_FAILED`
- `DIARIZATION_FAILED`
- `TASK_NOT_FOUND`
- `MEMORY_OVERFLOW`
- `INTERNAL_ERROR`

## 13. 实时会议纪要（1-pass + 2-pass 双路识别）

### 13.1 功能概述

基于 FunASR paraformer-zh-streaming 的流式双路识别方案，专为实时会议纪要场景优化：

- **1-pass 流式识别**：实时输出滚动字幕，用于实时展示
- **2-pass 全局修正**：满足触发条件时执行全局推理，输出最终纪要存档

### 13.2 架构说明

```
音频流输入
    │
    ▼
┌─────────────────────────────────────────┐
│           FSMN-VAD 静音检测              │
│         （断句、分句、状态判断）           │
└─────────────────────────────────────────┘
    │
    ├── 语音状态 ──→ 缓存音频 + 1-pass流式识别
    │                    │
    │                    ▼
    │            实时字幕展示（1-pass结果）
    │
    └── 静音状态 ──→ 检查触发条件
                          │
                          ├── VAD静音超时(0.5s) → 2-pass
                          ├── 长语音(6s) → 2-pass
                          └── 会话结束 → 2-pass兜底
                                    │
                                    ▼
                            最终纪要存档（2-pass结果）
```

### 13.3 2-pass 触发时机（会议场景标准）

| 触发机制 | 触发条件 | 业务意义 |
|---------|---------|---------|
| VAD静音超时 | 连续静音 ≥ 500ms | 自然断句，用户说话停顿/换气 |
| 长语音强制分片 | 单人连续说话 ≥ 6000ms | 防止ASR注意力漂移，保证识别质量 |
| 会话结束兜底 | WebSocket关闭/音频结束 | 确保所有缓存内容被识别存档 |

### 13.4 核心参数配置

```python
# 流式识别 chunk 配置（FunASR 官方推荐）
CHUNK_SIZE = [0, 10, 5]  # [left, middle, right]
# - left_chunks=0:  不回看左侧历史
# - middle_chunks=10: 每块 600ms (10 × 60ms)
# - right_chunks=5:  前瞻 300ms (5 × 60ms)

# 音频参数
DEFAULT_SAMPLE_RATE = 16000  # 必须16kHz单声道float32

# 2-pass 触发参数
VAD_SILENCE_TIMEOUT_MS = 500   # 静音超时 0.5s
MAX_SPEECH_DURATION_MS = 6000  # 长语音分片 6s
```

### 13.5 WebSocket 接口

连接地址：`ws://127.0.0.1:8000/api/v1/realtime-meeting/ws`

**客户端发送消息：**

```javascript
// 1. 创建会议会话
{"type": "start", "chunk_size": [0, 10, 5], "sample_rate": 16000}

// 2. 发送音频数据（二进制 PCM，16kHz, 16位, 单声道）
// 直接发送 ArrayBuffer

// 3. 结束会议
{"type": "end"}
```

**服务端返回消息：**

```javascript
// 会话创建成功
{"type": "started", "session_id": "xxx"}

// 1-pass 实时结果（用于展示）
{"type": "realtime", "segment_id": 0, "text": "新增文字", "full_text": "完整文字"}

// 2-pass 触发通知
{"type": "twopass_trigger", "trigger": "vad_silence", "message": "..."}

// 2-pass 最终结果（用于存档）
{"type": "final_segment", "segment_id": 0, "speaker": "spk0", "text": "最终文字"}

// 会议结束
{"type": "finished", "session_id": "xxx", "full_text": "...", "segments": [...]}
```

### 13.6 REST 接口

```bash
# 测试接口
POST /api/v1/realtime-meeting/test

# 获取会话信息
GET /api/v1/realtime-meeting/sessions/{session_id}

# 获取最终纪要
GET /api/v1/realtime-meeting/sessions/{session_id}/transcript
```

### 13.7 工程落地避坑清单

1. **音频格式必须正确**
   - 16kHz 单声道 float32
   - PCM 16位小端序
   - 避免采样率不匹配导致识别失败

2. **VAD 参数调优**
   - 会议场景：静音超时建议 300-500ms
   - 访谈场景：可适当延长至 800ms
   - 过短会导致句子被过度切分
   - 过长会导致一句话识别不完整

3. **长语音强制分片**
   - 必须实现，**禁止省略**
   - 超过6s的连续语音必须强制2-pass
   - 防止ASR注意力机制漂移导致错误累积

4. **2-pass 执行规范**
   - `input=None` + `is_final=True`
   - 执行后必须重置 ASR cache
   - 防止跨句子识别污染

5. **区分1-pass和2-pass用途**
   - 1-pass结果：仅用于实时展示
   - 2-pass结果：作为最终会议纪要存档
   - 禁止将1-pass结果作为正式记录

6. **会话关闭处理**
   - WebSocket断开时必须执行最后的2-pass
   - 确保缓存中的音频全部被识别
   - 避免数据丢失

7. **内存管理**
   - 长时间运行时定期清理会话
   - 监控 pending_audio_buffer 大小
   - 避免内存泄漏

### 13.8 依赖安装

```bash
# FunASR 及相关依赖
pip install funasr>=1.2.7
pip install modelscope>=1.17.1
pip install numpy>=1.26.0
pip install torch>=2.2.0
pip install torchaudio>=2.2.0

# WebSocket 客户端测试
pip install websocket-client
```

### 13.9 客户端示例（Python）

```python
import asyncio
import websockets
import numpy as np
import struct

async def test_realtime_meeting():
    uri = "ws://127.0.0.1:8000/api/v1/realtime-meeting/ws"

    async with websockets.connect(uri) as ws:
        # 1. 创建会话
        await ws.send('{"type": "start"}')
        resp = await ws.recv()
        print(f"会话创建: {resp}")

        # 2. 读取音频文件并发送
        import soundfile as sf
        audio, sr = sf.read("test.wav", dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # 转换为 PCM 16位
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # 分块发送（每秒一块）
        chunk_size = 16000 * 2  # 1秒音频
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await ws.send(chunk)

            # 接收识别结果
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=1.0)
                print(f"识别结果: {resp}")
            except asyncio.TimeoutError:
                pass

        # 3. 结束会话
        await ws.send('{"type": "end"}')
        final = await ws.recv()
        print(f"最终纪要: {final}")

asyncio.run(test_realtime_meeting())
```

## 14. 生产建议

- 建议使用 `gunicorn + uvicorn workers` 或容器化部署
- 长音频建议走异步任务接口
- 并发较高时，建议将任务状态与结果落到数据库或对象存储
- 若要增强跨分片说话人一致性，可进一步接入说话人嵌入比对逻辑
