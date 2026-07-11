# Triton Inference Server YOLOv8 Client
from backend.app.config.config import settings
from backend.app.shared.logging import logger


class TritonInferenceClient:
    def __init__(self):
        self.server_url = settings.TRITON_SERVER_URL
        self.model_name = settings.YOLO_MODEL_NAME

    def detect_objects(self, frame) -> list:
        """
        Sends frame to Triton Server running YOLOv8 (TensorRT).
        Returns detected objects with class, bounding box coordinates, and confidence.
        """
        logger.debug("Sending frame to Triton inference server", url=self.server_url)
        # Mock detection output
        # Format: [{"class": "car", "bbox": [x1, y1, x2, y2], "confidence": 0.95}]
        return [{"class": "car", "bbox": [100, 150, 250, 300], "confidence": 0.92}]


triton_client = TritonInferenceClient()
