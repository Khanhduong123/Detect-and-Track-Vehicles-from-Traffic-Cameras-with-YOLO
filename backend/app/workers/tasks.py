# Async Task Broker / Queue worker
from backend.app.shared.logging import logger
from backend.app.workers.extractor import FrameExtractor


async def process_video_upload_job(video_id: str, file_path: str):
    """
    Simulates a background task queue worker job pulling video tasks,
    extracting frames, and passing them to inference.
    """
    logger.info("Task worker pulled job", video_id=video_id, file=file_path)

    extractor = FrameExtractor(video_path=file_path)
    frames = extractor.extract_frames(
        sampling_rate=5
    )  # e.g. adaptive sampling every 5 frames

    logger.info(
        "Adaptive sampling completed", frames_count=len(frames), video_id=video_id
    )

    # After frame extraction, tasks are sent to AI Engine (Triton/ByteTrack)
    # trigger inference pipeline...
    return True
