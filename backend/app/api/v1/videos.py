# Video endpoints
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.shared.logging import logger
from backend.app.shared.state import jobs_status
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


@router.get("/status/{video_id}")
async def get_video_status(video_id: str):
    """
    Retrieves the processing status, progress percentage, and statistics of a video job.
    """
    if video_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_status[video_id]


@router.get("/download/{video_id}")
async def download_processed_video(video_id: str, background_tasks: BackgroundTasks):
    """
    Downloads/streams the annotated output video, and schedules cleanup of the file afterward.
    """
    file_path = f"/tmp/processed_{video_id}"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Processed video file not found or still processing",
        )

    def remove_file(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Cleaned up temporary processed video file", path=path)
        except Exception as e:
            logger.error(
                "Failed to clean up temporary processed video file",
                error=str(e),
                path=path,
            )

    background_tasks.add_task(remove_file, file_path)

    return FileResponse(
        file_path, media_type="video/mp4", filename=f"processed_{video_id}.mp4"
    )
