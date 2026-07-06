"""Tests for evaluate_map50_95.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from evaluate_map50_95 import (
    generate_plots,
    main,
    parse_arguments,
    print_comparison_tables,
    run_iou_analysis,
)


def test_parse_arguments_defaults() -> None:
    """Test that parse_arguments returns correct default values."""
    with patch("sys.argv", ["evaluate_map50_95.py"]):
        args = parse_arguments()
        assert args.data == "dataset/dataset.yaml"
        assert args.split == "test"
        assert args.batch == 16
        assert args.imgsz == 640
        assert args.weights is None
        assert args.output_dir == "."


def test_parse_arguments_custom() -> None:
    """Test that parse_arguments correctly parses custom values."""
    with patch(
        "sys.argv",
        [
            "evaluate_map50_95.py",
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
            "--output-dir",
            "results_out",
        ],
    ):
        args = parse_arguments()
        assert args.data == "custom_data.yaml"
        assert args.split == "val"
        assert args.device == "cpu"
        assert args.batch == 32
        assert args.imgsz == 320
        assert args.weights == "custom_weights.pt"
        assert args.output_dir == "results_out"


@patch("evaluate_map50_95.os.path.exists")
def test_run_iou_analysis_not_exist(mock_exists: MagicMock) -> None:
    """Test run_iou_analysis when the weights file does not exist."""
    mock_exists.return_value = False
    args = MagicMock()
    result = run_iou_analysis("MockModel", "nonexistent.pt", args)
    assert result == []


def _setup_mock_yolo_and_results(mock_exists: MagicMock, mock_yolo: MagicMock) -> tuple:
    """Helper to setup mocks for run_iou_analysis success tests."""
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

    mock_val_results = MagicMock()
    mock_val_results.names = {0: "motorbike", 1: "car"}
    mock_val_results.box = MagicMock()
    mock_val_results.box.all_ap = np.full((2, 10), 0.8)
    mock_model.val.return_value = mock_val_results

    return args, mock_model


def _assert_results(result: list) -> None:
    """Helper to verify results dictionaries content."""
    assert len(result) == 3  # 1 overall row + 2 class rows

    expected = [
        {
            "Model": "YOLOv8m",
            "Class": "all",
            "AP@50": 0.8,
            "AP@95": 0.8,
            "mAP50-95": 0.8,
        },
        {
            "Model": "YOLOv8m",
            "Class": "motorbike",
            "AP@50": 0.8,
            "mAP50-95": 0.8,
        },
    ]

    for idx, expected_dict in enumerate(expected):
        actual_dict = result[idx]
        for key, val in expected_dict.items():
            assert actual_dict[key] == val


@patch("evaluate_map50_95.YOLO")
@patch("evaluate_map50_95.os.path.exists")
def test_run_iou_analysis_success(
    mock_exists: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_iou_analysis when evaluation succeeds."""
    args, mock_model = _setup_mock_yolo_and_results(mock_exists, mock_yolo)

    result = run_iou_analysis("YOLOv8m", "weights/yolov8m.pt", args)

    _assert_results(result)

    mock_model.val.assert_called_once_with(
        data="dataset/dataset.yaml",
        split="test",
        batch=16,
        imgsz=640,
        device="cpu",
        plots=False,
    )


@patch("evaluate_map50_95.YOLO")
@patch("evaluate_map50_95.os.path.exists")
def test_run_iou_analysis_exception(
    mock_exists: MagicMock,
    mock_yolo: MagicMock,
) -> None:
    """Test run_iou_analysis handles exceptions during validation gracefully."""
    mock_exists.return_value = True
    args = MagicMock()
    mock_yolo.side_effect = RuntimeError("Runtime Error")

    result = run_iou_analysis("YOLOv8m", "weights/yolov8m.pt", args)
    assert result == []


def test_print_comparison_tables_empty() -> None:
    """Test print_comparison_tables with empty dataframe does not crash."""
    print_comparison_tables(pd.DataFrame())


def test_print_comparison_tables_populated() -> None:
    """Test print_comparison_tables prints correctly with data."""
    data = [
        {
            "Model": "YOLOv8m",
            "Class": "all",
            "AP@50": 0.85,
            "AP@55": 0.82,
            "AP@60": 0.80,
            "AP@65": 0.78,
            "AP@70": 0.75,
            "AP@75": 0.70,
            "AP@80": 0.65,
            "AP@85": 0.60,
            "AP@90": 0.50,
            "AP@95": 0.40,
            "mAP50-95": 0.685,
        },
        {
            "Model": "YOLOv8m",
            "Class": "car",
            "AP@50": 0.85,
            "AP@55": 0.82,
            "AP@60": 0.80,
            "AP@65": 0.78,
            "AP@70": 0.75,
            "AP@75": 0.70,
            "AP@80": 0.65,
            "AP@85": 0.60,
            "AP@90": 0.50,
            "AP@95": 0.40,
            "mAP50-95": 0.685,
        },
    ]
    df = pd.DataFrame(data)
    print_comparison_tables(df)


@patch("evaluate_map50_95.plt.savefig")
@patch("evaluate_map50_95.plt.show")
def test_generate_plots(mock_show: MagicMock, mock_savefig: MagicMock) -> None:
    """Test generate_plots generates and saves or shows figures."""
    # Test empty dataframe
    generate_plots(pd.DataFrame(), "out_dir")
    mock_savefig.assert_not_called()
    mock_show.assert_not_called()

    # Test populated dataframe
    data = [
        {
            "Model": "YOLOv8m",
            "Class": "all",
            "AP@50": 0.85,
            "AP@55": 0.82,
            "AP@60": 0.80,
            "AP@65": 0.78,
            "AP@70": 0.75,
            "AP@75": 0.70,
            "AP@80": 0.65,
            "AP@85": 0.60,
            "AP@90": 0.50,
            "AP@95": 0.40,
            "mAP50-95": 0.685,
        },
        {
            "Model": "YOLOv8m",
            "Class": "car",
            "AP@50": 0.85,
            "AP@55": 0.82,
            "AP@60": 0.80,
            "AP@65": 0.78,
            "AP@70": 0.75,
            "AP@75": 0.70,
            "AP@80": 0.65,
            "AP@85": 0.60,
            "AP@90": 0.50,
            "AP@95": 0.40,
            "mAP50-95": 0.685,
        },
    ]
    df = pd.DataFrame(data)

    # Test case 1: With output_dir (should call savefig and not show)
    generate_plots(df, "out_dir")
    assert mock_savefig.call_count == 2
    mock_show.assert_not_called()

    # Reset mocks
    mock_savefig.reset_mock()
    mock_show.reset_mock()

    # Test case 2: Without output_dir (should call show and not savefig)
    generate_plots(df, None)
    mock_savefig.assert_not_called()
    assert mock_show.call_count == 2


@patch("evaluate_map50_95.generate_plots")
@patch("evaluate_map50_95.print_comparison_tables")
@patch("evaluate_map50_95.pd.DataFrame.to_csv")
@patch("evaluate_map50_95.run_iou_analysis")
@patch("evaluate_map50_95.parse_arguments")
def test_main_pipeline(
    mock_parse_args: MagicMock,
    mock_run_iou: MagicMock,
    mock_to_csv: MagicMock,
    mock_print_tables: MagicMock,
    mock_gen_plots: MagicMock,
) -> None:
    """Test the main execution pipeline."""
    args = MagicMock(weights=None, output_dir="out_dir")
    mock_parse_args.return_value = args
    mock_run_iou.return_value = [
        {"Model": "YOLOv8m", "Class": "all", "AP@50": 0.8, "mAP50-95": 0.8}
    ]

    main()

    mock_run_iou.assert_called()
    mock_to_csv.assert_called_once()
    mock_print_tables.assert_called_once()
    mock_gen_plots.assert_called_once()
