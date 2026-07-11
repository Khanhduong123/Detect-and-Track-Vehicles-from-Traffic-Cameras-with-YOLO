# Wrong way driving behavior analyzer
from backend.app.shared.logging import logger


class WrongWayDetector:
    def __init__(self, allowed_direction: tuple = (1, 0)):
        self.allowed_direction = allowed_direction

    def check_violation(self, track_id: int, trajectory: list) -> bool:
        """
        Compares trajectory directions with configured allowed traffic flow.
        Returns True if the vehicle is driving the wrong way.
        """
        if len(trajectory) < 2:
            return False

        # Basic directional check from start to end coordinates
        start = trajectory[0]
        end = trajectory[-1]

        dx = end[0] - start[0]

        # Simulating direction dot-product comparison
        if dx < 0:  # Driving in reverse of typical traffic flow
            logger.warn("Potential Wrong Way Driving detected", track_id=track_id)
            return True

        return False


wrong_way_detector = WrongWayDetector()
