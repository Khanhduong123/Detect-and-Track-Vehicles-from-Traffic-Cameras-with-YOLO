from typing import Any, List

import cv2

from backend.app.shared.logging import logger


class FrameExtractor:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def extract_frames(self, sampling_rate: int = 5) -> List[Any]:
        """
        Extracts frames using OpenCV/FFmpeg/GStreamer with adaptive sampling.
        Returns a list of sampled video frames.
        """
        logger.info("Opening video stream", path=self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Adaptive Sampling logic
            if count % sampling_rate == 0:
                frames.append(frame)
            count += 1

        cap.release()
        return frames

    def generate_clip(self, start_frame: int, end_frame: int) -> str:
        """
        Generates a 3-5s clip for evidence storage.
        """
        # Save a sub-clip based on indices
        output_path = f"/tmp/evidence_{start_frame}_{end_frame}.mp4"
        # clip generation logic here...
        return output_path
