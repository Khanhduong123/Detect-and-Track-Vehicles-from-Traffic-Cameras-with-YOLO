import os

import cv2
import numpy as np
import pytest

from backend.app.workers.extractor import FrameExtractor


@pytest.fixture
def temp_video_file(tmp_path):
    """
    Creates a temporary video file with:
    - 30 static frames (black image)
    - 30 moving frames (black image with a moving white circle)
    Total: 60 frames at 30 FPS.
    """
    video_file_path = os.path.join(tmp_path, "test_video.mp4")
    fps = 30.0
    width, height = 320, 240

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_file_path, fourcc, fps, (width, height))

    # 30 static frames
    for _ in range(30):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)

    # 30 moving frames (draw a moving circle)
    for i in range(30):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (50 + i * 5, 120), 15, (255, 255, 255), -1)
        out.write(frame)

    out.release()
    yield video_file_path


def test_metadata_extraction(temp_video_file):
    extractor = FrameExtractor(temp_video_file)
    meta = extractor.get_metadata()

    assert meta["fps"] == 30.0
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["frame_count"] == 60
    assert abs(meta["duration"] - 2.0) < 1e-5


def test_missing_video_file():
    with pytest.raises(FileNotFoundError):
        FrameExtractor("non_existent_file.mp4")


def test_interval_sampling(temp_video_file):
    extractor = FrameExtractor(temp_video_file)

    # Test sampling every 10 frames
    frames = extractor.extract_frames(sampling_rate=10, sampling_mode="interval")
    # Expected frame indices: 0, 10, 20, 30, 40, 50 -> 6 frames
    assert len(frames) == 6


def test_fps_sampling(temp_video_file):
    extractor = FrameExtractor(temp_video_file)

    # Video is 30 FPS. Target FPS 3.0 -> sampling rate should be round(30/3) = 10
    frames = extractor.extract_frames(sampling_mode="fps", target_fps=3.0)
    assert len(frames) == 6


def test_motion_sampling(temp_video_file):
    extractor = FrameExtractor(temp_video_file)

    # In motion mode, the extractor should sample:
    # 1. The first frame (index 0).
    # 2. Since frames 1-29 are static (no motion), it should only sample them if max_interval is reached.
    #    With max_interval=20, it should sample at count=20.
    # 3. From frame 30 onwards, there is motion, so it should sample frequently.
    frames = extractor.extract_frames(
        sampling_mode="motion",
        motion_threshold=0.01,
        max_interval=20,
        min_interval=2,
    )

    # The total number of sampled frames should be greater than the static-only baseline
    # (which would be 60 / 20 = 3 frames).
    assert len(frames) > 3
    # Check that frames are returned as numpy arrays
    assert isinstance(frames[0], np.ndarray)


def test_generate_clip(temp_video_file, tmp_path):
    extractor = FrameExtractor(temp_video_file)
    output_clip_path = os.path.join(tmp_path, "test_clip.mp4")

    # Generate a clip from frame 15 to 45 (31 frames total)
    res_path = extractor.generate_clip(
        start_frame=15, end_frame=45, output_path=output_clip_path
    )

    assert os.path.exists(res_path)
    assert res_path == output_clip_path

    # Read back the generated clip to verify its metadata
    cap = cv2.VideoCapture(res_path)
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert width == 320
    assert height == 240
    assert frame_count == 31
