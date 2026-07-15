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
        dy = end[1] - start[1]

        # Filter out negligible movements to avoid false alerts from bounding box jitter
        displacement = (dx**2 + dy**2) ** 0.5
        if displacement < 15.0:  # Minimum pixel displacement
            return False

        # Calculate dot-product of movement vector and configured allowed direction
        dot_product = dx * self.allowed_direction[0] + dy * self.allowed_direction[1]

        if dot_product < 0:  # Driving in reverse of typical traffic flow
            logger.warning("Potential Wrong Way Driving detected", track_id=track_id)
            return True

        return False


wrong_way_detector = WrongWayDetector()
