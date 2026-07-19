import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "voice-doc-service"
    app_env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    engine: Literal["funasr", "whisper"] = "funasr"
    enable_gpu: bool = True
    force_cpu: bool = False

    model_cache_dir: Path = Path("./models")
    temp_dir: Path = Path("./data/tmp")
    result_dir: Path = Path("./data/results")
    task_dir: Path = Path("./data/tasks")
    log_dir: Path = Path("./logs")

    max_audio_duration_seconds: int = 14400
    max_upload_size_mb: int = 512
    sample_rate: int = 16000
    audio_chunk_seconds: int = 1800
    batch_size_seconds: int = 300
    denoise_enabled: bool = True

    supported_audio_extensions: Union[str, list[str]] = Field(
        default_factory=lambda: [".mp3", ".wav", ".flac", ".m4a", ".webm", ".ogg", ".aac"]
    )

    funasr_model: str = "iic/SenseVoiceSmall"
    funasr_streaming_model: str = "paraformer-zh-streaming"
    funasr_vad_model: str = "fsmn-vad"
    funasr_punc_model: Optional[str] = None  # SenseVoice 不需要标点模型
    funasr_spk_model: str = "cam++"
    funasr_hub: str = "ms"
    funasr_vad_kwargs: Optional[dict] = None  # 可选：{"max_single_segment_time": 30000}

    whisper_model: str = "small"
    whisper_language: Optional[str] = None
    whisper_download_root: Path = Path("./models/whisper")
    whisper_compute_type: Literal["float16", "int8", "float32"] = "float16"

    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    pyannote_auth_token: Optional[str] = None
    pyannote_num_speakers: Optional[int] = None

    task_worker_count: int = 2

    # 模型预加载配置
    preload_models_on_startup: bool = True  # 启动时预加载模型

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator(
        "model_cache_dir",
        "temp_dir",
        "result_dir",
        "task_dir",
        "log_dir",
        "whisper_download_root",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, value: Union[str, Path]) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("supported_audio_extensions", mode="before")
    @classmethod
    def _parse_extensions_before(cls, value: Union[str, list[str]]) -> Union[str, list[str]]:
        return value

    @field_validator("supported_audio_extensions", mode="after")
    @classmethod
    def _parse_extensions_after(cls, value: Union[str, list[str]]) -> list[str]:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return [item.lower() for item in value]

    @field_validator(
        "whisper_language",
        "pyannote_auth_token",
        "pyannote_num_speakers",
        mode="before"
    )
    @classmethod
    def _empty_str_to_none(cls, value: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def ensure_directories(self) -> None:
        for path in (
            self.model_cache_dir,
            self.temp_dir,
            self.result_dir,
            self.task_dir,
            self.log_dir,
            self.whisper_download_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.model_cache_dir))
        os.environ.setdefault("MODELSCOPE_CACHE", str(self.model_cache_dir))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
