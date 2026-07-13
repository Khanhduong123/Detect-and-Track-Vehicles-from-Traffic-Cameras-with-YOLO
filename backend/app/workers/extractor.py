import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.app.shared.logging import logger


class FrameExtractor:
    def __init__(self, video_path: str):
        """
        Initializes the FrameExtractor with the video path.
        Validates the path if it is a local file.
        """
        self.video_path = video_path
        is_stream = video_path.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        if not is_stream and not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extracts video metadata including FPS, dimensions, total frame count, and duration.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        cap.release()

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration": duration,
        }

    def _preprocess_frame(self, frame: Any) -> Any:
        """
        Converts a frame to grayscale, resizes it, and applies Gaussian blur.
        This prepares the frame for motion diff calculation.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def _extract_interval_or_fps(
        self, cap: cv2.VideoCapture, sampling_rate: int
    ) -> List[Any]:
        """
        Reads and returns frames sampled at a fixed interval rate.
        """
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % sampling_rate == 0:
                frames.append(frame)
            count += 1
        return frames

    def _extract_motion(
        self,
        cap: cv2.VideoCapture,
        motion_threshold: float,
        max_interval: int,
        min_interval: int,
    ) -> List[Any]:
        """
        Reads and returns frames sampled adaptively using pixel motion change threshold.
        """
        frames = []
        count = 0
        prev_gray = None
        last_sampled_count = -max_interval

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if prev_gray is None:
                frames.append(frame)
                prev_gray = self._preprocess_frame(frame)
                last_sampled_count = count
            else:
                frames_since_sample = count - last_sampled_count

                if frames_since_sample >= max_interval:
                    frames.append(frame)
                    prev_gray = self._preprocess_frame(frame)
                    last_sampled_count = count
                elif frames_since_sample >= min_interval:
                    curr_gray_blur = self._preprocess_frame(frame)
                    diff = cv2.absdiff(prev_gray, curr_gray_blur)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    motion_ratio = np.sum(thresh == 255) / thresh.size

                    if motion_ratio >= motion_threshold:
                        frames.append(frame)
                        prev_gray = curr_gray_blur
                        last_sampled_count = count

            count += 1
        return frames

    def extract_frames(
        self,
        sampling_rate: int = 5,
        sampling_mode: str = "interval",
        target_fps: Optional[float] = None,
        motion_threshold: float = 0.02,
        max_interval: int = 30,
        min_interval: int = 2,
    ) -> List[Any]:
        """
        Extracts frames using OpenCV with adaptive sampling.
        Returns a list of sampled video frames (as numpy arrays).

        Supported sampling modes:
        - "interval": Samples every `sampling_rate` frames.
        - "fps": Dynamically downsamples based on `target_fps` relative to source FPS.
        - "motion": Adaptive motion-based sampling. Detects changes in the video using
                    frame difference and dynamically adjusts sampling.
        """
        logger.info(
            "Opening video stream for frame extraction",
            path=self.video_path,
            mode=sampling_mode,
        )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {self.video_path}")

        try:
            if sampling_mode == "fps":
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                if target_fps is None or target_fps <= 0:
                    target_fps = 2.0
                # Calculate dynamic interval to achieve target_fps
                sampling_rate = max(1, round(fps / target_fps))
                logger.info(
                    "Dynamic FPS sampling",
                    source_fps=fps,
                    target_fps=target_fps,
                    calculated_rate=sampling_rate,
                )

            if sampling_mode == "motion":
                frames = self._extract_motion(
                    cap,
                    motion_threshold=motion_threshold,
                    max_interval=max_interval,
                    min_interval=min_interval,
                )
            else:
                frames = self._extract_interval_or_fps(cap, sampling_rate)
        finally:
            cap.release()

        logger.info(
            "Frame extraction completed",
            extracted_frames=len(frames),
        )
        return frames

    def _setup_video_writer(
        self, output_path: str, fps: float, width: int, height: int
    ) -> cv2.VideoWriter:
        """
        Creates and returns a VideoWriter object, falling back to XVID if mp4v fails.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.warning("mp4v codec failed to open, trying XVID fallback")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return out

    def _write_clip_frames(
        self,
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        start_frame: int,
        end_frame: int,
    ) -> None:
        """
        Sets position to start_frame and writes frames up to end_frame.
        """
        seek_success = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if not seek_success or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start_frame:
            # Fallback to sequential read if seek is unsupported or fails
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(start_frame):
                ret, _ = cap.read()
                if not ret:
                    break
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

    def generate_clip(
        self, start_frame: int, end_frame: int, output_path: Optional[str] = None
    ) -> str:
        """
        Generates a sub-clip for evidence storage.
        Clamps frames to valid ranges, uses mp4v codec by default, and falls back if needed.
        """
        if not output_path:
            output_path = f"/tmp/evidence_{start_frame}_{end_frame}.mp4"

        # Ensure directory of output_path exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {self.video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0:
                start_frame = max(0, min(start_frame, total_frames - 1))
                end_frame = max(start_frame, min(end_frame, total_frames - 1))
            else:
                if start_frame < 0:
                    start_frame = 0
                if end_frame < start_frame:
                    end_frame = start_frame

            logger.info(
                "Generating clip",
                start_frame=start_frame,
                end_frame=end_frame,
                output_path=output_path,
            )

            out = self._setup_video_writer(output_path, fps, width, height)
            try:
                self._write_clip_frames(cap, out, start_frame, end_frame)
            finally:
                out.release()
        finally:
            cap.release()
        return output_path
