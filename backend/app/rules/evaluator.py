# Event Detection coordinator
from backend.app.rules.speed import speed_analyst
from backend.app.rules.wrong_way import wrong_way_detector
from backend.app.services.alert import alert_service
from backend.app.services.storage import storage_service
from backend.app.shared.logging import logger


class EventEvaluator:
    async def evaluate_track_events(
        self, track_id: int, trajectory: list, camera_id: str
    ):
        """
        Coordinates behavior evaluation, triggers evidence video generation & alerts
        upon rule violation detection.
        """
        is_wrong_way = wrong_way_detector.check_violation(track_id, trajectory)
        speed = speed_analyst.calculate_speed(track_id, trajectory)

        if is_wrong_way or speed > speed_analyst.speed_limit:
            violation_type = "wrong_way" if is_wrong_way else "speeding"
            logger.info(
                "Violation event confirmed!", track_id=track_id, type=violation_type
            )

            # Generate subclip and upload evidence to Cloud Storage
            # clip_path = frame_extractor.generate_clip(...)
            evidence_url = await storage_service.upload_evidence_clip(
                file_path="/tmp/evidence.mp4",
                filename=f"evidence_{camera_id}_{track_id}.mp4",
            )

            # Dispatch realtime alert
            await alert_service.trigger_violation_alert(
                violation_id=999, violation_type=violation_type, camera_id=camera_id
            )

            return {
                "violation_detected": True,
                "type": violation_type,
                "speed": speed,
                "evidence_url": evidence_url,
            }

        return {"violation_detected": False}


event_evaluator = EventEvaluator()
