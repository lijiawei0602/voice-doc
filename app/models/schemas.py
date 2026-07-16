from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    speaker: str = "spk0"
    start_ms: int
    end_ms: int
    text: str
    chunk_index: int = 0


class SpeakerSummary(BaseModel):
    speaker: str
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    task_id: str
    engine: Literal["funasr", "whisper"]
    device: str
    source: str
    language: Optional[str] = None
    duration_seconds: float
    text: str
    segments: list[TranscriptSegment]
    speakers: list[SpeakerSummary]
    result_path: Optional[Path] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPathRequest(BaseModel):
    paths: list[str]


class BatchPathItem(BaseModel):
    source: str
    result: Optional[TranscriptResult] = None
    error: Optional[str] = None


class AsyncBatchResponse(BaseModel):
    task_ids: list[str]


class TaskStatusPayload(BaseModel):
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    source: Optional[str] = None
    result_path: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[TranscriptResult] = None


# 流式语音识别相关 Schema

class StreamingSegment(BaseModel):
    """流式识别片段"""
    segment_id: int
    text: str
    start_ms: int = 0
    end_ms: int = 0
    is_final: bool = False
    speaker: str = "spk0"


class StreamingResult(BaseModel):
    """流式识别结果"""
    session_id: str
    text: str
    segments: list[StreamingSegment] = Field(default_factory=list)
    language: Optional[str] = None


class StreamingStartResponse(BaseModel):
    """流式识别开始响应"""
    session_id: str
    message: str = "Streaming session started"
