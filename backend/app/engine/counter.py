# Traffic Volume Counting (ID-based)
from backend.app.shared.logging import logger


class TrafficCounter:
    def __init__(self):
        self.counted_ids = set()

    def process_tracks(self, tracked_objects: list) -> int:
        """
        Maintains unique count of objects passing coordinate lines.
        Returns the accumulated vehicle count.
        """
        for obj in tracked_objects:
            track_id = obj.get("track_id")
            if track_id and track_id not in self.counted_ids:
                # In real scenario, check if track intersects a counting gate/line
                self.counted_ids.add(track_id)
                logger.info("New vehicle counted", track_id=track_id)

        return len(self.counted_ids)


traffic_counter = TrafficCounter()
