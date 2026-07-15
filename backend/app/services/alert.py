# Alert Service
from backend.app.shared.logging import logger


class AlertService:
    def trigger_violation_alert(
        self, violation_id: int, violation_type: str, camera_id: str
    ) -> bool:
        """
        Sends real-time violation alert trigger.
        Integrates with Notification system / WebSockets / Webhooks.
        """
        logger.info(
            "ALERT TRIGGERED: Violation detected!",
            violation_id=violation_id,
            violation_type=violation_type,
            camera_id=camera_id,
        )
        # Implement alert delivery logic (e.g. HTTP post, WebSocket push)
        return True


alert_service = AlertService()
