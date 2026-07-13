# Async Task Broker / Queue worker
import datetime
import time
from typing import List, Set, Tuple

import cv2
import numpy as np

from backend.app.db.models import ViolationEvent
from backend.app.engine.tracker import tracker_service
from backend.app.engine.triton import triton_client
from backend.app.rules.speed import SpeedAnalyst
from backend.app.rules.wrong_way import WrongWayDetector
from backend.app.services.alert import alert_service
from backend.app.shared.logging import logger
from backend.app.shared.state import jobs_status, mock_db_store


def draw_annotations(
    frame: np.ndarray,
    track_id: int,
    bbox: List[float],
    cls_name: str,
    speed: float,
    is_speeding: bool,
    is_wrong_way: bool,
    history: List[List[float]],
) -> None:
    """Draws bounding boxes, speeds, and trajectories on the frame."""
    x1, y1, x2, y2 = map(int, bbox)
    if is_speeding:
        color = (0, 0, 255)  # Red
        thickness = 3
    elif is_wrong_way:
        color = (0, 140, 255)  # Orange
        thickness = 3
    else:
        colors = {
            "car": (240, 180, 56),
            "truck": (160, 230, 80),
            "bus": (200, 100, 240),
            "motorcycle": (60, 220, 240),
        }
        color = colors.get(cls_name, (200, 200, 200))
        thickness = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"#{track_id} {cls_name} | {speed:.1f} km/h"
    if is_speeding:
        label += " (SPEEDING)"
    if is_wrong_way:
        label += " (WRONG WAY)"
    cv2.putText(
        frame,
        label,
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        lineType=cv2.LINE_AA,
    )

    if len(history) >= 2:
        pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA
        )


async def _handle_violations(
    track_id: int,
    cls_name: str,
    bbox: List[float],
    speed: float,
    is_speeding: bool,
    is_wrong_way: bool,
    flagged_speed_ids: Set[int],
    flagged_wrong_ids: Set[int],
) -> None:
    """Handles triggering alerts and appending events to the mock DB."""
    if is_speeding and track_id not in flagged_speed_ids:
        flagged_speed_ids.add(track_id)
        event = ViolationEvent(
            id=len(mock_db_store) + 1,
            violation_type="speeding",
            timestamp=datetime.datetime.utcnow(),
            metadata_json={
                "speed": speed,
                "bbox": bbox,
                "class": cls_name,
            },
            track_id=track_id,
            camera_id="cam_01",
        )
        mock_db_store.append(event)
        await alert_service.trigger_violation_alert(
            violation_id=track_id,
            violation_type="speeding",
            camera_id="cam_01",
        )

    if is_wrong_way and track_id not in flagged_wrong_ids:
        flagged_wrong_ids.add(track_id)
        event = ViolationEvent(
            id=len(mock_db_store) + 1,
            violation_type="wrong_way",
            timestamp=datetime.datetime.utcnow(),
            metadata_json={
                "bbox": bbox,
                "class": cls_name,
            },
            track_id=track_id,
            camera_id="cam_01",
        )
        mock_db_store.append(event)
        await alert_service.trigger_violation_alert(
            violation_id=track_id,
            violation_type="wrong_way",
            camera_id="cam_01",
        )


async def _process_single_frame(
    frame: np.ndarray,
    video_id: str,
    fps: float,
    speed_analyst: SpeedAnalyst,
    wrong_way_detector: WrongWayDetector,
    flagged_speed_ids: Set[int],
    flagged_wrong_ids: Set[int],
    unique_counted_ids: Set[int],
) -> None:
    """Orchestrates frame object detection, tracking, violation checking, and drawing."""
    detections = triton_client.detect_objects(frame, conf_threshold=0.45)

    tracker_input = [
        {"bbox": d["bbox"], "confidence": d["confidence"], "class": d["class"]}
        for d in detections
    ]

    tracked_objects = tracker_service.update_tracks(tracker_input, video_id=video_id)

    for obj in tracked_objects:
        track_id = obj["track_id"]
        trajectory = obj["history"]
        cls_name = obj["class"]
        bbox = obj["bbox"]

        unique_counted_ids.add(track_id)

        speed = speed_analyst.calculate_speed(track_id, trajectory, fps=fps)
        is_speeding = speed > 60.0
        is_wrong_way = wrong_way_detector.check_violation(track_id, trajectory)

        await _handle_violations(
            track_id=track_id,
            cls_name=cls_name,
            bbox=bbox,
            speed=speed,
            is_speeding=is_speeding,
            is_wrong_way=is_wrong_way,
            flagged_speed_ids=flagged_speed_ids,
            flagged_wrong_ids=flagged_wrong_ids,
        )

        draw_annotations(
            frame=frame,
            track_id=track_id,
            bbox=bbox,
            cls_name=cls_name,
            speed=speed,
            is_speeding=is_speeding,
            is_wrong_way=is_wrong_way,
            history=trajectory,
        )


def _init_job_status(video_id: str) -> None:
    """Initializes status dictionary entry for the job."""
    jobs_status[video_id] = {
        "status": "processing",
        "progress": 0.0,
        "total_vehicles": 0,
        "speed_violations": 0,
        "wrong_way_violations": 0,
        "fps": 0.0,
    }


def _update_job_status(
    video_id: str,
    frame_count: int,
    total_frames: int,
    t_start: float,
    unique_counted_ids: Set[int],
    flagged_speed_ids: Set[int],
    flagged_wrong_ids: Set[int],
) -> None:
    """Calculates FPS and progress percentage to update status entry."""
    elapsed = time.time() - t_start
    live_fps = frame_count / elapsed if elapsed > 0 else 0.0
    progress = min(float(frame_count) / total_frames * 100, 100.0)

    jobs_status[video_id] = {
        "status": "processing",
        "progress": progress,
        "total_vehicles": len(unique_counted_ids),
        "speed_violations": len(flagged_speed_ids),
        "wrong_way_violations": len(flagged_wrong_ids),
        "fps": live_fps,
    }


def _init_video_writer(
    cap: cv2.VideoCapture, video_id: str
) -> Tuple[cv2.VideoWriter, float, int]:
    """Retrieves properties from video source and prepares VideoWriter output."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = fps if fps > 0 else 30.0
    total_frames = total_frames if total_frames > 0 else 100

    out_path = f"/tmp/processed_{video_id}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    return out, fps, total_frames


async def process_video_upload_job(video_id: str, file_path: str):
    """
    Background job that extracts frames, queries Triton, tracks vehicles,
    evaluates rules, generates alerts, writes an annotated output video,
    and updates progress stats.
    """
    logger.info("Task worker pulled job", video_id=video_id, file=file_path)
    _init_job_status(video_id)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error("Failed to open input video file", file=file_path)
        jobs_status[video_id]["status"] = "failed"
        return False

    out, fps, total_frames = _init_video_writer(cap, video_id)

    # Initialize rule processors
    speed_analyst = SpeedAnalyst(speed_limit_kmh=60.0, pixels_per_meter=15.0)
    wrong_way_detector = WrongWayDetector(allowed_direction=(1.0, 0.0))

    flagged_speed_ids: Set[int] = set()
    flagged_wrong_ids: Set[int] = set()
    unique_counted_ids: Set[int] = set()

    frame_count = 0
    t_start = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            await _process_single_frame(
                frame=frame,
                video_id=video_id,
                fps=fps,
                speed_analyst=speed_analyst,
                wrong_way_detector=wrong_way_detector,
                flagged_speed_ids=flagged_speed_ids,
                flagged_wrong_ids=flagged_wrong_ids,
                unique_counted_ids=unique_counted_ids,
            )

            out.write(frame)
            _update_job_status(
                video_id=video_id,
                frame_count=frame_count,
                total_frames=total_frames,
                t_start=t_start,
                unique_counted_ids=unique_counted_ids,
                flagged_speed_ids=flagged_speed_ids,
                flagged_wrong_ids=flagged_wrong_ids,
            )

        jobs_status[video_id]["status"] = "completed"
        jobs_status[video_id]["progress"] = 100.0
        logger.info(
            "Video processing completed",
            video_id=video_id,
            out_file=f"/tmp/processed_{video_id}",
        )
        return True

    except Exception as e:
        logger.error(
            "Error during video processing job", error=str(e), video_id=video_id
        )
        jobs_status[video_id]["status"] = "failed"
        return False
    finally:
        cap.release()
        out.release()
