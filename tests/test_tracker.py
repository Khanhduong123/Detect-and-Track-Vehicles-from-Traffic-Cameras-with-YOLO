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
