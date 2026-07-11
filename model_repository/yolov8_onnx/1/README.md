# YOLOv8 ONNX Model Placeholder

Please place your exported YOLOv8 ONNX model file in this directory and name it `model.onnx`.

## Steps to export the model:
1. Export the YOLOv8 model to ONNX:
   ```bash
   yolo export model=yolov8n.pt format=onnx imgsz=640
   ```
2. Move the exported `yolov8n.onnx` file into this directory and rename it to `model.onnx`:
   ```
   model_repository/yolov8_onnx/1/model.onnx
   ```
