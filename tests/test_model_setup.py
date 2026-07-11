import os
import re


def test_model_exists():
    model_path = "model_repository/yolov8_onnx/1/model.onnx"
    assert os.path.exists(model_path), "YOLOv8 ONNX model file does not exist!"
    assert (
        os.path.getsize(model_path) > 10 * 1024 * 1024
    ), "YOLOv8 ONNX model file is too small!"


def test_triton_config():
    config_path = "model_repository/yolov8_onnx/config.pbtxt"
    assert os.path.exists(config_path), "Triton config.pbtxt file does not exist!"

    with open(config_path, "r") as f:
        content = f.read()

    # Check that model name is configured correctly
    assert 'name: "yolov8_onnx"' in content

    # Check that output dimensions match the custom model (1 batch, 8 classes + boxes, 8400 candidates)
    assert (
        re.search(r"dims:\s*\[\s*1,\s*8,\s*8400\s*\]", content) is not None
    ), "Output dims in config.pbtxt should be [ 1, 8, 8400 ]!"
