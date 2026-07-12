from backend.app.rules.speed import SpeedAnalyst


def test_calculate_speed_insufficient_data():
    analyst = SpeedAnalyst(speed_limit_kmh=60.0, pixels_per_meter=15.0)
    assert analyst.calculate_speed(1, []) == 0.0
    assert analyst.calculate_speed(1, [[100, 100]]) == 0.0


def test_calculate_speed_normal():
    # 15 pixels displacement, pixels_per_meter = 15.0 -> 1.0 meter.
    # 2 frames -> 1 frame interval -> 1/30 seconds (at 30 fps).
    # speed = 1.0 / (1/30) = 30 m/s = 108 km/h.
    analyst = SpeedAnalyst(speed_limit_kmh=120.0, pixels_per_meter=15.0)
    trajectory = [[100, 100], [115, 100]]
    speed = analyst.calculate_speed(track_id=1, trajectory=trajectory, fps=30.0)
    assert abs(speed - 108.0) < 1e-5


def test_calculate_speed_speeding():
    # Speed is 108 km/h, which is above the default speed limit of 60.0 km/h.
    analyst = SpeedAnalyst(speed_limit_kmh=60.0, pixels_per_meter=15.0)
    trajectory = [[100, 100], [115, 100]]
    speed = analyst.calculate_speed(track_id=2, trajectory=trajectory, fps=30.0)
    assert abs(speed - 108.0) < 1e-5
