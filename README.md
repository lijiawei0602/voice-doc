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
- 支持 `mp3/wav/flac/m4a`
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
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
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

## 13. 生产建议

- 建议使用 `gunicorn + uvicorn workers` 或容器化部署
- 长音频建议走异步任务接口
- 并发较高时，建议将任务状态与结果落到数据库或对象存储
- 若要增强跨分片说话人一致性，可进一步接入说话人嵌入比对逻辑
