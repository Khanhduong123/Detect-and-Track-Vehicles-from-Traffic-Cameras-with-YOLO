# Triton Inference Server YOLOv8 Client
import cv2
import numpy as np
import tritonclient.http as httpclient

from backend.app.config.config import settings
from backend.app.shared.logging import logger


class TritonInferenceClient:
    def __init__(self):
        self.server_url = settings.TRITON_SERVER_URL
        self.model_name = settings.YOLO_MODEL_NAME

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resizes, normalizes, and transposes the frame for model input."""
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        # Transpose to CHW and add batch dimension
        img_input = np.transpose(img_input, (2, 0, 1))
        return np.expand_dims(img_input, axis=0)

    def _postprocess(
        self,
        output_data: np.ndarray,
        img_w: int,
        img_h: int,
        conf_threshold: float,
    ) -> list:
        """Processes Triton output tensor, applies NMS, and returns formatted detections."""
        output = np.squeeze(output_data, axis=0)  # (8, 8400)

        # Extract scores and find best class
        scores = output[4:, :]  # (4, 8400)
        class_ids = np.argmax(scores, axis=0)  # (8400,)
        confidences = np.max(scores, axis=0)  # (8400,)

        mask = confidences >= conf_threshold
        if not np.any(mask):
            return []

        cand_boxes = output[:4, mask]  # (4, N)
        cand_confs = confidences[mask]  # (N,)
        cand_classes = class_ids[mask]  # (N,)

        boxes = []
        class_names = ["car", "truck", "bus", "motorcycle"]

        # Calculate coordinates
        for i in range(cand_boxes.shape[1]):
            cx, cy, w, h = cand_boxes[:, i]
            x1 = (cx - w / 2.0) * (img_w / 640.0)
            y1 = (cy - h / 2.0) * (img_h / 640.0)
            x2 = (cx + w / 2.0) * (img_w / 640.0)
            y2 = (cy + h / 2.0) * (img_h / 640.0)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])

        indices = cv2.dnn.NMSBoxes(boxes, cand_confs.tolist(), conf_threshold, 0.45)

        detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else indices
            for idx in flat_indices:
                cls_idx = int(cand_classes[idx])
                detections.append(
                    {
                        "class": (
                            class_names[cls_idx]
                            if cls_idx < len(class_names)
                            else "unknown"
                        ),
                        "bbox": boxes[idx],
                        "confidence": float(cand_confs[idx]),
                    }
                )
        return detections

    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.45) -> list:
        """
        Sends frame to Triton Server running YOLOv8 (TensorRT).
        Returns detected objects with class, bounding box coordinates, and confidence.
        """
        logger.debug("Sending frame to Triton inference server", url=self.server_url)
        try:
            img_h, img_w = frame.shape[:2]
            img_input = self._preprocess(frame)

            # Resolve Triton HTTP port (Triton uses 8000 for HTTP, 8001 for gRPC)
            url = self.server_url
            if "8001" in url:
                url = url.replace("8001", "8000")

            client = httpclient.InferenceServerClient(url=url)

            # Setup input and output
            inputs = [httpclient.InferInput("images", img_input.shape, "FP32")]
            inputs[0].set_data_from_numpy(img_input)

            outputs = [httpclient.InferRequestedOutput("output0")]

            # Execute inference
            response = client.infer(self.model_name, inputs, outputs=outputs)
            output_data = response.as_numpy("output0")

            return self._postprocess(output_data, img_w, img_h, conf_threshold)

        except Exception as e:
            logger.warning(
                "Triton inference failed, returning empty detection set",
                error=str(e),
                url=self.server_url,
            )
            return []


triton_client = TritonInferenceClient()
