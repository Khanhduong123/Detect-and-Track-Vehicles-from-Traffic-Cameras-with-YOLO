# Speed Analyst
from backend.app.shared.logging import logger


class SpeedAnalyst:
    def __init__(self, speed_limit_kmh: float = 60.0, pixels_per_meter: float = 15.0):
        self.speed_limit = speed_limit_kmh
        self.pixels_per_meter = pixels_per_meter

    def calculate_speed(
        self, track_id: int, trajectory: list, fps: float = 30.0
    ) -> float:
        """
        Calculates the speed of a vehicle based on pixel displacement over frame time.
        """
        if len(trajectory) < 2:
            return 0.0

        start, end = trajectory[0], trajectory[-1]
        pixel_distance = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5

        # distance in meters
        meters = pixel_distance / self.pixels_per_meter
        seconds = len(trajectory) / fps

        speed_mps = meters / seconds
        speed_kmh = speed_mps * 3.6

        if speed_kmh > self.speed_limit:
            logger.warn(
                "Potential Speeding detected", track_id=track_id, speed=speed_kmh
            )

        return speed_kmh


speed_analyst = SpeedAnalyst()
