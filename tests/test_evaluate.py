"""Tests for evaluate.py."""

from unittest.mock import MagicMock, patch

import pandas as pd

from evaluate import (
    generate_plots,
    main,
    parse_arguments,
    print_comparison_table,
    run_single_evaluation,
)


def test_parse_arguments_defaults() -> None:
    """Test that parse_arguments returns correct default values."""
    with patch("sys.argv", ["evaluate.py"]):
        args = parse_arguments()
        assert args.data == "dataset/dataset.yaml"
        assert args.split == "test"
        assert args.batch == 16
        assert args.imgsz == 640
        assert args.weights is None


def test_parse_arguments_custom() -> None:
    """Test that parse_arguments correctly parses custom values."""
    with patch(
        "sys.argv",
        [
            "evaluate.py",
            "--data",
            "custom_data.yaml",
            "--split",
            "val",
            "--device",
            "cpu",
            "--batch",
            "32",
            "--imgsz",
            "320",
            "--weights",
            "custom_weights.pt",
        ],
    ):
        args = parse_arguments()
        assert args.data == "custom_data.yaml"
        assert args.split == "val"
        assert args.device == "cpu"
        assert args.batch == 32
        assert args.imgsz == 320
        assert args.weights == "custom_weights.pt"


@patch("evaluate.os.path.exists")
def test_run_single_evaluation_not_exist(mock_exists: MagicMock) -> None:
    """Test run_single_evaluation when the weights file does not exist."""
    mock_exists.return_value = False
    args = MagicMock()
    result = run_single_evaluation("MockModel", "nonexistent.pt", args)
    assert result == {}


def _create_mock_val_results() -> MagicMock:
    """Helper to create a populated MagicMock for val results."""
    mock_val_results = MagicMock()
    mock_val_results.results_dict = {
        "metrics/precision(B)": 0.8,
        "metrics/recall(B)": 0.75,
        "metrics/mAP50(B)": 0.85,
        "metrics/mAP50-95(B)": 0.6,
        "fitness": 0.6,
    }
    mock_val_results.speed = {
        "inference": 12.5,
        "preprocess": 0.5,
        "postprocess": 1.0,
    }
    return mock_val_results


@patch("evaluate.YOLO")
@patch("evaluate.os.path.exists")
def test_run_single_evaluation_success(
    mock_exists: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_single_evaluation when evaluation succeeds."""
    mock_exists.return_value = True
    args = MagicMock(
        data="dataset/dataset.yaml",
        split="test",
        batch=16,
        imgsz=640,
        device="cpu",
    )

    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    mock_model.val.return_value = _create_mock_val_results()

    result = run_single_evaluation("YOLOv8m", "weights/yolov8m.pt", args)

    expected = {
        "Model": "YOLOv8m",
        "Weights": "weights/yolov8m.pt",
        "precision(B)": 0.8,
        "recall(B)": 0.75,
        "mAP50(B)": 0.85,
        "mAP50-95(B)": 0.6,
        "fitness": 0.6,
        "Inference_ms": 12.5,
        "Preprocess_ms": 0.5,
        "Postprocess_ms": 1.0,
    }
    assert result == expected

    mock_model.val.assert_called_once_with(
        data="dataset/dataset.yaml",
        split="test",
        batch=16,
        imgsz=640,
        device="cpu",
        plots=False,
    )


@patch("evaluate.YOLO")
@patch("evaluate.os.path.exists")
def test_run_single_evaluation_exception(
    mock_exists: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_single_evaluation handles exceptions from YOLO initialization or validation."""
    mock_exists.return_value = True
    args = MagicMock()
    mock_yolo.side_effect = RuntimeError("Failed to load weights")

    result = run_single_evaluation("YOLOv8m", "weights/yolov8m.pt", args)
    assert result == {}


def test_print_comparison_table_empty() -> None:
    """Test print_comparison_table with empty results."""
    # Ensure it runs without exception
    print_comparison_table([])


def test_print_comparison_table_populated() -> None:
    """Test print_comparison_table with populated results."""
    results = [
        {
            "Model": "YOLOv8m",
            "precision(B)": 0.8,
            "recall(B)": 0.7,
            "mAP50(B)": 0.85,
            "mAP50-95(B)": 0.6,
            "fitness": 0.6,
            "Inference_ms": 12.0,
        }
    ]
    # Ensure it runs without exception
    print_comparison_table(results)


@patch("evaluate.plt.savefig")
def test_generate_plots(mock_savefig: MagicMock) -> None:
    """Test generate_plots generates a matplotlib bar chart."""
    # Empty df check
    generate_plots(pd.DataFrame(), "plot.png")
    mock_savefig.assert_not_called()

    # Populated df check
    df = pd.DataFrame(
        [
            {
                "Model": "YOLOv8m",
                "mAP50(B)": 0.85,
                "mAP50-95(B)": 0.6,
                "Split": "test",
            }
        ]
    )
    generate_plots(df, "plot.png")
    mock_savefig.assert_called_once_with("plot.png", dpi=300)


@patch("evaluate.generate_plots")
@patch("evaluate.pd.DataFrame.to_csv")
@patch("evaluate.print_comparison_table")
@patch("evaluate.run_single_evaluation")
@patch("evaluate.parse_arguments")
def test_main_pipeline(
    mock_parse_args: MagicMock,
    mock_run_eval: MagicMock,
    mock_print_table: MagicMock,
    mock_to_csv: MagicMock,
    mock_gen_plots: MagicMock,
) -> None:
    """Test main function pipeline execution."""
    args = MagicMock(weights=None, split="test")
    mock_parse_args.return_value = args
    mock_run_eval.side_effect = [
        {"Model": "YOLOv8m", "mAP50(B)": 0.8},
        {"Model": "YOLOv9m", "mAP50(B)": 0.82},
        {"Model": "YOLOv8m_Pre", "mAP50(B)": 0.03},
        {"Model": "YOLOv9m_Pre", "mAP50(B)": 0.04},
    ]

    main()

    assert mock_run_eval.call_count == 4
    mock_print_table.assert_called_once()
    mock_to_csv.assert_called_once()
    mock_gen_plots.assert_called_once()


@patch("evaluate.generate_plots")
@patch("evaluate.pd.DataFrame.to_csv")
@patch("evaluate.print_comparison_table")
@patch("evaluate.run_single_evaluation")
@patch("evaluate.parse_arguments")
def test_main_pipeline_custom_weights(
    mock_parse_args: MagicMock,
    mock_run_eval: MagicMock,
    mock_print_table: MagicMock,
    mock_to_csv: MagicMock,
    mock_gen_plots: MagicMock,
) -> None:
    """Test main function pipeline execution when custom weights are specified."""
    args = MagicMock(weights="custom_weights.pt", split="test")
    mock_parse_args.return_value = args
    mock_run_eval.return_value = {"Model": "custom_weights.pt", "mAP50(B)": 0.8}

    main()

    mock_run_eval.assert_called_once_with(
        "custom_weights.pt", "custom_weights.pt", args
    )
    mock_print_table.assert_called_once()
    mock_to_csv.assert_called_once()
    mock_gen_plots.assert_called_once()


@patch("evaluate.run_single_evaluation")
@patch("evaluate.parse_arguments")
def test_main_pipeline_no_results(
    mock_parse_args: MagicMock,
    mock_run_eval: MagicMock,
) -> None:
    """Test main function pipeline when no models were successfully evaluated."""
    args = MagicMock(weights="custom_weights.pt", split="test")
    mock_parse_args.return_value = args
    mock_run_eval.return_value = {}

    # Ensure it handles no results gracefully and returns early
    main()
