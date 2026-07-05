"""Tests for utils.py."""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from utils import (
    evaluate_model,
    get_model_name,
    load_yaml_config,
    load_yolo_model,
    merge_configs,
    register_tqdm_callbacks,
    setup_mlflow_resume,
)


def test_load_yaml_config_not_exists() -> None:
    """Test load_yaml_config when path does not exist."""
    assert load_yaml_config("nonexistent_file.yaml") == {}


def test_load_yaml_config_valid(tmp_path) -> None:
    """Test load_yaml_config with a valid config file."""
    config_file = tmp_path / "config.yaml"
    data = {"epochs": 10, "batch": 8, "device": "cpu"}
    with open(config_file, "w") as f:
        yaml.safe_dump(data, f)

    loaded = load_yaml_config(str(config_file))
    assert loaded == data


@patch("utils.open")
def test_load_yaml_config_exception(mock_open: MagicMock) -> None:
    """Test load_yaml_config handles exceptions gracefully."""
    mock_open.side_effect = OSError("Access denied")
    assert load_yaml_config("hyp.yaml") == {}


def test_merge_configs_all_overrides() -> None:
    """Test merge_configs with CLI overrides and YAML config."""
    args = argparse.Namespace(
        cfg="hyp.yaml",
        model="yolov9m",
        data="custom.yaml",
        epochs=15,
        batch=4,
        imgsz=320,
        device="cpu",
        workers=2,
        project="runs",
        name="test_run",
        optimizer="Adam",
        lr0=0.002,
        box=5.0,
        cls=0.3,
        fraction=0.8,
        resume=True,
        mosaic=0.1,
        mixup=0.2,
        copy_paste=0.3,
        degrees=15.0,
        scale=0.6,
        fliplr=0.7,
    )
    cfg_data = {"epochs": 50, "batch": 32}
    merged = merge_configs(args, cfg_data)

    assert merged["model"] == "yolov9m"
    assert merged["epochs"] == 15  # CLI takes precedence
    assert merged["batch"] == 4  # CLI takes precedence
    assert merged["resume"] is True


def test_merge_configs_defaults() -> None:
    """Test merge_configs fallbacks to defaults when no CLI or YAML is provided."""
    args = argparse.Namespace(resume=False)
    merged = merge_configs(args, {})
    assert merged["model"] == "yolov8m.pt"
    assert merged["epochs"] == 100
    assert merged["batch"] == 16
    assert merged["resume"] is False


@patch("utils.mlflow")
def test_setup_mlflow_resume_existing(mock_mlflow: MagicMock) -> None:
    """Test setup_mlflow_resume when an existing run matches name."""
    mock_runs = MagicMock()
    mock_runs.empty = False
    mock_runs.iloc = [{"run_id": "run_123"}]
    mock_mlflow.search_runs.return_value = mock_runs

    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

    setup_mlflow_resume("run_name_example", "exp_name")

    assert os.environ.get("MLFLOW_RUN_ID") == "run_123"
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.search_runs.assert_called_once()


@patch("utils.mlflow")
def test_setup_mlflow_resume_not_found(mock_mlflow: MagicMock) -> None:
    """Test setup_mlflow_resume when no matching run is found."""
    mock_runs = MagicMock()
    mock_runs.empty = True
    mock_mlflow.search_runs.return_value = mock_runs

    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

    setup_mlflow_resume("run_name_example", "exp_name")
    assert "MLFLOW_RUN_ID" not in os.environ


@patch("utils.YOLO")
@patch("utils.os.path.exists")
def test_load_yolo_model_base(mock_exists: MagicMock, mock_yolo: MagicMock) -> None:
    """Test load_yolo_model loads pretrained model when resume is False."""
    mock_exists.return_value = False
    load_yolo_model("yolov8m.pt", resume=False, project="runs", name="exp")
    mock_yolo.assert_called_once_with("yolov8m.pt")


@patch("utils.YOLO")
@patch("utils.os.path.exists")
def test_load_yolo_model_local_weights(
    mock_exists: MagicMock, mock_yolo: MagicMock
) -> None:
    """Test load_yolo_model loads from weights/ folder when file exists."""
    # First call to exists (for raw name) is False, second call (for weights_path) is True
    mock_exists.side_effect = lambda path: path == os.path.join("weights", "yolov8m.pt")
    load_yolo_model("yolov8m.pt", resume=False, project="runs", name="exp")
    mock_yolo.assert_called_once_with(os.path.join("weights", "yolov8m.pt"))


@patch("utils.YOLO")
@patch("utils.os.path.exists")
def test_load_yolo_model_resume_explicit(
    mock_exists: MagicMock, mock_yolo: MagicMock
) -> None:
    """Test load_yolo_model loads explicit last.pt path."""
    mock_exists.return_value = True
    load_yolo_model("runs/exp/weights/last.pt", resume=True, project="runs", name="exp")
    mock_yolo.assert_called_once_with("runs/exp/weights/last.pt")


@patch("utils.YOLO")
@patch("utils.os.path.exists")
def test_load_yolo_model_resume_search(
    mock_exists: MagicMock, mock_yolo: MagicMock
) -> None:
    """Test load_yolo_model searches and finds last.pt checkpoint."""
    # os.path.exists returns True only for the second candidate
    mock_exists.side_effect = lambda path: "runs/detect/runs/detect" in path
    load_yolo_model("yolov8m.pt", resume=True, project="runs", name="exp")
    mock_yolo.assert_called_once_with("runs/detect/runs/detect/exp/weights/last.pt")


@patch("utils.YOLO")
@patch("utils.glob.glob")
@patch("utils.os.path.getmtime")
@patch("utils.os.path.exists")
def test_load_yolo_model_resume_glob(
    mock_exists: MagicMock,
    mock_getmtime: MagicMock,
    mock_glob: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test load_yolo_model glob fallback sorts matches by mtime descending."""
    mock_exists.side_effect = lambda path: path == "new_path/exp/weights/last.pt"
    mock_glob.return_value = [
        "old_path/exp/weights/last.pt",
        "new_path/exp/weights/last.pt",
    ]
    mock_getmtime.side_effect = lambda path: 1000 if "old_path" in path else 2000

    load_yolo_model("yolov8m.pt", resume=True, project="runs", name="exp")
    mock_yolo.assert_called_once_with("new_path/exp/weights/last.pt")


def test_load_yolo_model_resume_not_found() -> None:
    """Test load_yolo_model raises FileNotFoundError when checkpoint is missing."""
    with pytest.raises(FileNotFoundError):
        load_yolo_model(
            "yolov8m.pt", resume=True, project="nonexistent", name="nonexistent"
        )


def test_get_model_name() -> None:
    """Test get_model_name formats and defaults correctly."""
    assert get_model_name({"model": "yolov8m"}) == "yolov8m.pt"
    assert get_model_name({"model": "yolov9m.pt"}) == "yolov9m.pt"
    assert get_model_name({}) == "yolov8m.pt"


@patch("utils.YOLO")
def test_evaluate_model(mock_yolo_class: MagicMock) -> None:
    """Test evaluate_model triggers validation on the test split."""
    mock_model = MagicMock()
    mock_val_results = MagicMock()
    mock_val_results.results_dict = {"metrics/mAP50": 0.9}
    mock_model.val.return_value = mock_val_results

    evaluate_model(mock_model)
    mock_model.val.assert_called_once_with(split="test")


@patch("utils.YOLO")
def test_register_tqdm_callbacks(mock_yolo_class: MagicMock) -> None:
    """Test register_tqdm_callbacks adds callbacks to the YOLO model."""
    mock_model = MagicMock()
    register_tqdm_callbacks(mock_model)

    assert mock_model.add_callback.call_count == 3
    # Check that callbacks are named appropriately
    calls = [call[0][0] for call in mock_model.add_callback.call_args_list]
    assert "on_train_epoch_start" in calls
    assert "on_train_batch_end" in calls
    assert "on_train_epoch_end" in calls
