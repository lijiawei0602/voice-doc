from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS
from app.core.logging import get_logger
from app.models.schemas import TaskStatusPayload, TranscriptResult
from app.services.pipeline.transcription_service import TranscriptionService
from app.utils.json_store import read_json, write_json

logger = get_logger(__name__)


class TaskManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.executor = ThreadPoolExecutor(max_workers=self.settings.task_worker_count)
        self.lock = Lock()

    def submit(self, source_path: Path, original_source: str) -> str:
        task_id = uuid4().hex
        payload = self._build_payload(
            task_id=task_id,
            status="pending",
            source=original_source,
        )
        self._save_task(payload)
        self.executor.submit(self._run_task, task_id, source_path, original_source)
        return task_id

    def get(self, task_id: str) -> TaskStatusPayload:
        task_file = self.settings.task_dir / f"{task_id}.json"
        if not task_file.exists():
            raise AppError(ERRORS["TASK_NOT_FOUND"])

        payload = read_json(task_file)
        return TaskStatusPayload.model_validate(payload)

    def _run_task(self, task_id: str, source_path: Path, original_source: str) -> None:
        with self.lock:
            pending = self.get(task_id).model_dump(mode="json")
            pending["status"] = "running"
            pending["updated_at"] = self._now()
            self._save_task(pending)

        try:
            service = TranscriptionService()
            result = service.transcribe_file_with_id(
                source_path=source_path,
                task_id=task_id,
                original_source=original_source,
            )
            completed = self._build_payload(
                task_id=task_id,
                status="completed",
                source=original_source,
                result_path=str(result.result_path) if result.result_path else None,
                result=result,
            )
            self._save_task(completed)
        except AppError as exc:
            logger.error("任务失败: task_id=%s code=%s message=%s", task_id, exc.detail.code, exc.detail.message)
            failed = self._build_payload(
                task_id=task_id,
                status="failed",
                source=original_source,
                error_code=exc.detail.code,
                error_message=exc.detail.message,
            )
            self._save_task(failed)
        except Exception as exc:
            logger.error("任务异常失败: task_id=%s error=%s", task_id, exc)
            failed = self._build_payload(
                task_id=task_id,
                status="failed",
                source=original_source,
                error_code=ERRORS["INTERNAL_ERROR"].code,
                error_message=str(exc),
            )
            self._save_task(failed)

    def _save_task(self, payload: dict) -> None:
        task_file = self.settings.task_dir / f"{payload['task_id']}.json"
        write_json(task_file, payload)

    def _build_payload(
        self,
        task_id: str,
        status: str,
        source: str | None,
        result_path: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        result: TranscriptResult | None = None,
    ) -> dict:
        old = {}
        task_file = self.settings.task_dir / f"{task_id}.json"
        if task_file.exists():
            old = read_json(task_file)

        return {
            "task_id": task_id,
            "status": status,
            "created_at": old.get("created_at", self._now()),
            "updated_at": self._now(),
            "source": source,
            "result_path": result_path,
            "error_code": error_code,
            "error_message": error_message,
            "result": result.model_dump(mode="json") if result else None,
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()


task_manager = TaskManager()
