from backend.app.engine.tracker import ByteTracker, tracker_service


def test_bbox_iou_overlap():
    box1 = [0.0, 0.0, 2.0, 2.0]
    box2 = [1.0, 1.0, 3.0, 3.0]
    assert abs(ByteTracker.bbox_iou(box1, box2) - 1.0 / 7.0) < 1e-5


def test_bbox_iou_no_overlap():
    box1 = [0.0, 0.0, 2.0, 2.0]
    box3 = [5.0, 5.0, 6.0, 6.0]
    assert ByteTracker.bbox_iou(box1, box3) == 0.0


def test_bytetracker_initial():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.5)
    dets = [{"bbox": [10.0, 10.0, 20.0, 20.0], "confidence": 0.9, "class": "car"}]
    tracks = tracker.update(dets)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].bbox == [10.0, 10.0, 20.0, 20.0]


def test_bytetracker_update():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.5)
    dets = [{"bbox": [10.0, 10.0, 20.0, 20.0], "confidence": 0.9, "class": "car"}]
    tracker.update(dets)

    dets_frame2 = [
        {"bbox": [11.0, 11.0, 21.0, 21.0], "confidence": 0.85, "class": "car"}
    ]
    tracks = tracker.update(dets_frame2)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].bbox == [11.0, 11.0, 21.0, 21.0]


def test_bytetracker_low_score_association():
    tracker = ByteTracker(track_thresh=0.6, match_thresh=0.5)
    dets = [{"bbox": [10.0, 10.0, 20.0, 20.0], "confidence": 0.9, "class": "car"}]
    tracker.update(dets)

    dets_frame2 = [
        {"bbox": [11.0, 11.0, 21.0, 21.0], "confidence": 0.4, "class": "car"}
    ]
    tracks = tracker.update(dets_frame2)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].score == 0.4


def test_tracker_service_video1():
    dets = [{"bbox": [100, 100, 200, 200], "confidence": 0.95, "class": "truck"}]
    res1 = tracker_service.update_tracks(dets, video_id="video_1")

    assert len(res1) == 1
    assert res1[0]["track_id"] == 1
    assert res1[0]["class"] == "truck"


def test_tracker_service_video2():
    dets = [{"bbox": [100, 100, 200, 200], "confidence": 0.95, "class": "truck"}]
    res2 = tracker_service.update_tracks(dets, video_id="video_2")

    assert len(res2) == 1
    assert res2[0]["track_id"] == 1
    assert res2[0]["class"] == "truck"


def test_tracker_bottom_center_and_smoothing():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.5)

    # 1. Test bottom-center coordinate extraction
    dets = [{"bbox": [100, 100, 200, 300], "confidence": 0.9, "class": "car"}]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    # Bottom center of [100, 100, 200, 300] is ((100+200)/2, 300) = (150.0, 300.0)
    assert tracks[0].center == (150.0, 300.0)
    assert tracks[0].history == [[150.0, 300.0]]

    # 2. Test smoothing and history sliding window
    # Update track 12 times with shifting boxes
    for i in range(1, 13):
        # Shift bbox x by 10 pixels each time
        # Raw bottom center will be (150.0 + i*10, 300.0)
        x_shift = i * 10
        bbox_new = [100 + x_shift, 100, 200 + x_shift, 300]
        dets_new = [{"bbox": bbox_new, "confidence": 0.9, "class": "car"}]
        tracks = tracker.update(dets_new)

    assert len(tracks) == 1
    # Check that history length is capped at 10
    assert len(tracks[0].history) == 10

    # Check that the last coordinate is smoothed (moving average of last 5 raw points)
    # Raw history coordinates for last 5 frames (i=8 to 12):
    # i=8: 150 + 80 = 230
    # i=9: 150 + 90 = 240
    # i=10: 150 + 100 = 250
    # i=11: 150 + 110 = 260
    # i=12: 150 + 120 = 270
    # Average x: (230 + 240 + 250 + 260 + 270) / 5 = 250.0
    assert abs(tracks[0].history[-1][0] - 250.0) < 1e-5
    assert abs(tracks[0].history[-1][1] - 300.0) < 1e-5
