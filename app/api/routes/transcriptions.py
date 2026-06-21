from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.core.response import success_response
from app.models.schemas import BatchPathItem, BatchPathRequest
from app.services.pipeline.transcription_service import TranscriptionService
from app.utils.files import save_upload_file, stage_local_file

router = APIRouter(prefix="/api/v1/transcriptions", tags=["transcriptions"])


@router.post("/upload")
async def transcribe_upload(file: UploadFile = File(...)) -> dict:
    stored_path = await save_upload_file(file)
    service = TranscriptionService()
    result = service.transcribe_file(stored_path, original_source=file.filename or str(stored_path))
    return success_response(result.model_dump(mode="json"))


@router.post("/batch")
async def transcribe_batch(request: BatchPathRequest) -> dict:
    service = TranscriptionService()
    items: list[BatchPathItem] = []
    for source in request.paths:
        try:
            stored_path = stage_local_file(source)
            result = service.transcribe_file(stored_path, original_source=source)
            items.append(BatchPathItem(source=source, result=result))
        except Exception as exc:
            items.append(BatchPathItem(source=source, error=str(exc)))
    return success_response({"items": [item.model_dump(mode="json") for item in items]})
