from backend.app.rules.wrong_way import WrongWayDetector


def test_wrong_way_insufficient_data():
    detector = WrongWayDetector(allowed_direction=(1, 0))
    assert not detector.check_violation(1, [])
    assert not detector.check_violation(1, [[100, 100]])


def test_wrong_way_default():
    detector = WrongWayDetector(allowed_direction=(1, 0))

    # Moving in the allowed direction (along x)
    trajectory_ok = [[100, 100], [110, 100]]
    assert not detector.check_violation(track_id=1, trajectory=trajectory_ok)

    # Moving opposite to the allowed direction
    trajectory_wrong = [[100, 100], [90, 100]]
    assert detector.check_violation(track_id=2, trajectory=trajectory_wrong)


def test_wrong_way_vertical():
    # Traffic is allowed to move downwards (along positive y-axis)
    detector = WrongWayDetector(allowed_direction=(0, 1))

    # Moving downward
    trajectory_ok = [[100, 100], [100, 110]]
    assert not detector.check_violation(track_id=3, trajectory=trajectory_ok)

    # Moving upward (opposite)
    trajectory_wrong = [[100, 100], [100, 90]]
    assert detector.check_violation(track_id=4, trajectory=trajectory_wrong)
