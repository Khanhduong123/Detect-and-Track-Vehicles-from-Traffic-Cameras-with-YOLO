# Video endpoints
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from backend.app.shared.logging import logger
from backend.app.workers.tasks import process_video_upload_job

router = APIRouter()


@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Video upload API endpoint. Receives file and enqueues frame extraction/inference job.
    """
    logger.info("Video upload received", filename=file.filename)

    # Save uploaded file to temporary directory/storage
    safe_filename = f"{uuid.uuid4()}_{os.path.basename(file.filename or 'video.mp4')}"
    temp_path = f"/tmp/{safe_filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logger.error("Failed to write uploaded video file", error=str(e))
        raise HTTPException(status_code=500, detail="File upload failed")

    # Enqueue task broker job
    background_tasks.add_task(
        process_video_upload_job, video_id=safe_filename, file_path=temp_path
    )

    return {
        "status": "success",
        "message": "Video uploaded successfully. Inference job scheduled in background.",
        "filename": file.filename,
        "video_id": safe_filename,
    }
