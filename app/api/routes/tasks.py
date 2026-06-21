from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.core.response import success_response
from app.models.schemas import AsyncBatchResponse, BatchPathRequest
from app.services.tasks.task_manager import task_manager
from app.utils.files import save_upload_file, stage_local_file

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("/upload")
async def submit_upload_task(file: UploadFile = File(...)) -> dict:
    stored_path = await save_upload_file(file)
    task_id = task_manager.submit(stored_path, file.filename or str(stored_path))
    return success_response({"task_id": task_id}, message="task submitted")


@router.post("/batch")
async def submit_batch_tasks(request: BatchPathRequest) -> dict:
    task_ids: list[str] = []
    for source in request.paths:
        stored_path = stage_local_file(source)
        task_ids.append(task_manager.submit(stored_path, source))
    payload = AsyncBatchResponse(task_ids=task_ids)
    return success_response(payload.model_dump(mode="json"), message="tasks submitted")


@router.get("/{task_id}")
def get_task_status(task_id: str) -> dict:
    payload = task_manager.get(task_id)
    return success_response(payload.model_dump(mode="json"))
