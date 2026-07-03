"""Tests for train.py."""

from unittest.mock import MagicMock, patch

from train import parse_args, run_training


def test_parse_args_defaults() -> None:
    """Test that parse_args returns correct default values."""
    args = parse_args([])
    expected = {
        "cfg": "hyp.yaml",
        "model": None,
        "data": None,
        "epochs": None,
        "batch": None,
        "imgsz": None,
        "device": None,
        "workers": None,
        "project": None,
        "name": None,
        "optimizer": None,
        "lr0": None,
        "box": None,
        "cls": None,
        "fraction": None,
        "resume": False,
        "mosaic": None,
        "mixup": None,
        "copy_paste": None,
        "degrees": None,
        "scale": None,
        "fliplr": None,
    }
    assert vars(args) == expected


def test_parse_args_custom() -> None:
    """Test that parse_args correctly parses custom values."""
    args = parse_args(
        [
            "--cfg",
            "custom_hyp.yaml",
            "--model",
            "yolov9m.pt",
            "--data",
            "custom_data.yaml",
            "--epochs",
            "50",
            "--batch",
            "32",
            "--imgsz",
            "320",
            "--device",
            "1",
            "--workers",
            "4",
            "--project",
            "custom_runs",
            "--name",
            "custom_name",
            "--optimizer",
            "SGD",
            "--lr0",
            "0.005",
            "--box",
            "10.0",
            "--cls",
            "0.8",
            "--fraction",
            "0.05",
            "--resume",
            "--mosaic",
            "0.5",
            "--mixup",
            "0.1",
            "--copy_paste",
            "0.2",
            "--degrees",
            "5.0",
            "--scale",
            "0.2",
            "--fliplr",
            "0.3",
        ]
    )
    expected = {
        "cfg": "custom_hyp.yaml",
        "model": "yolov9m.pt",
        "data": "custom_data.yaml",
        "epochs": 50,
        "batch": 32,
        "imgsz": 320,
        "device": "1",
        "workers": 4,
        "project": "custom_runs",
        "name": "custom_name",
        "optimizer": "SGD",
        "lr0": 0.005,
        "box": 10.0,
        "cls": 0.8,
        "fraction": 0.05,
        "resume": True,
        "mosaic": 0.5,
        "mixup": 0.1,
        "copy_paste": 0.2,
        "degrees": 5.0,
        "scale": 0.2,
        "fliplr": 0.3,
    }
    assert vars(args) == expected


@patch("train.YOLO")
@patch("train.torch.cuda.is_available")
@patch("train.os.path.exists")
def test_run_training_base(
    mock_exists: MagicMock,
    mock_cuda_available: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_training with base parameters when the config file does not exist."""
    mock_cuda_available.return_value = True
    # Make os.path.exists return False for the cfg to use fallbacks
    mock_exists.return_value = False
    args = parse_args([])

    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    mock_val_results = MagicMock()
    mock_val_results.results_dict = {"metrics/mAP50": 0.85}
    mock_model.val.return_value = mock_val_results

    run_training(args)

    mock_yolo.assert_called_once_with("yolov8m.pt")
    # Verify it trained with FALLBACK_DEFAULTS
    mock_model.train.assert_called_once()
    kwargs = mock_model.train.call_args[1]
    assert kwargs["data"] == "dataset/dataset.yaml"
    assert kwargs["epochs"] == 100
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["device"] == "0"


@patch("train.YOLO")
@patch("train.os.path.exists")
@patch("train.torch.cuda.is_available")
def test_run_training_resume(
    mock_cuda_available: MagicMock,
    mock_exists: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_training with resume active and checkpoint existing."""
    mock_cuda_available.return_value = False
    # mock exists to return False for hyp.yaml, but True for checkpoint weight
    mock_exists.side_effect = lambda path: "last.pt" in path
    args = parse_args(["--resume"])

    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    mock_val_results = MagicMock()
    mock_val_results.results_dict = {"metrics/mAP50": 0.85}
    mock_model.val.return_value = mock_val_results

    run_training(args)

    mock_yolo.assert_called_once_with("runs/detect/yolov8m_traffic/weights/last.pt")
    mock_model.train.assert_called_once()
    kwargs = mock_model.train.call_args[1]
    assert kwargs["resume"] is True


@patch("train.YOLO")
@patch("train.torch.cuda.is_available")
@patch("train.os.path.exists")
def test_run_training_custom_model(
    mock_exists: MagicMock,
    mock_cuda_available: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_training with a custom model name like yolov9m."""
    mock_cuda_available.return_value = False
    mock_exists.return_value = False
    args = parse_args(["--model", "yolov9m"])

    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    mock_val_results = MagicMock()
    mock_val_results.results_dict = {"metrics/mAP50": 0.85}
    mock_model.val.return_value = mock_val_results

    run_training(args)

    mock_yolo.assert_called_once_with("yolov9m.pt")
    mock_model.train.assert_called_once()
    kwargs = mock_model.train.call_args[1]
    assert "model" not in kwargs


@patch("train.YOLO")
@patch("train.torch.cuda.is_available")
@patch("train.os.path.exists")
def test_run_training_model_edge_cases(
    mock_exists: MagicMock,
    mock_cuda_available: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_training with non-string and non-standard model name extensions."""
    mock_cuda_available.return_value = False
    mock_exists.return_value = False

    # Mock merge_configs to return a dict with model: None
    with patch("train.merge_configs") as mock_merge:
        mock_merge.return_value = {
            "model": None,
            "device": "cpu",
            "resume": False,
            "project": "runs",
            "name": "exp",
        }
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_val_results = MagicMock()
        mock_val_results.results_dict = {"metrics/mAP50": 0.85}
        mock_model.val.return_value = mock_val_results

        run_training(parse_args([]))
        mock_yolo.assert_called_with("yolov8m.pt")

    # Mock merge_configs to return a dict with a yaml model configuration file
    with patch("train.merge_configs") as mock_merge:
        mock_yolo.reset_mock()
        mock_merge.return_value = {
            "model": "yolov8m.yaml",
            "device": "cpu",
            "resume": False,
            "project": "runs",
            "name": "exp",
        }
        run_training(parse_args([]))
        mock_yolo.assert_called_with("yolov8m.yaml")
