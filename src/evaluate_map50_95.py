"""Script to analyze YOLO model performance across the mAP50-95 IoU threshold range.

This script runs validation on a specified dataset split, extracts class-specific
and overall Average Precision (AP) at each IoU threshold (0.50 to 0.95 in 0.05 steps),
saves the detailed analysis to a CSV file, prints tables to the console,
and generates plots illustrating the performance curves.
"""

import argparse
import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

DEFAULT_MODELS = {
    "YOLOv8m (Custom)": "runs/detect/yolov8m_traffic-4/weights/best.pt",
    "YOLOv8m (Pretrained)": "yolov8m.pt",
}

IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for mAP50-95 range analysis.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze YOLO model range between mAP50 and mAP95."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/dataset.yaml",
        help="Path to dataset config YAML file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (val, test, or train).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (e.g. 0 or cpu).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to specific model weights. If provided, only this model is evaluated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save CSV and plot outputs.",
    )
    return parser.parse_args()


def _format_results(
    model_name: str,
    class_names: Dict[int, str],
    all_ap: np.ndarray,
) -> List[Dict[str, Any]]:
    """Format raw AP array into a list of dictionaries with proper type hinting."""
    results_data: List[Dict[str, Any]] = []

    # Calculate overall mAP at each IoU threshold (mean across classes)
    overall_ap_per_iou = np.mean(all_ap, axis=0)

    # 1. Add "all" (mean across classes) row
    overall_row: Dict[str, Any] = {
        "Model": model_name,
        "Class": "all",
    }
    for idx, thresh in enumerate(IOU_THRESHOLDS):
        overall_row[f"AP@{int(thresh * 100)}"] = float(overall_ap_per_iou[idx])
    # Add summary mAP50-95 (average over the 10 thresholds)
    overall_row["mAP50-95"] = float(np.mean(overall_ap_per_iou))
    results_data.append(overall_row)

    # 2. Add class-specific rows
    for class_id in range(all_ap.shape[0]):
        class_name = class_names.get(class_id, f"class_{class_id}")
        class_row: Dict[str, Any] = {
            "Model": model_name,
            "Class": class_name,
        }
        for idx, thresh in enumerate(IOU_THRESHOLDS):
            class_row[f"AP@{int(thresh * 100)}"] = float(all_ap[class_id, idx])
        class_row["mAP50-95"] = float(np.mean(all_ap[class_id, :]))
        results_data.append(class_row)

    return results_data


def _execute_validation(
    weights_path: str,
    args: argparse.Namespace,
) -> Any:
    """Helper to run the YOLO model validation loop."""
    try:
        model = YOLO(weights_path)
        return model.val(
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            plots=False,
        )
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None


def run_iou_analysis(
    model_name: str,
    weights_path: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Evaluate a single model and extract AP values at all IoU thresholds.

    Args:
        model_name: Friendly name of the model.
        weights_path: Path to the weights .pt file.
        args: Parsed command-line arguments.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing evaluation metrics
                              per class and overall.
    """
    if not os.path.exists(weights_path):
        print(f"Skipping {model_name}: weights file not found at '{weights_path}'")
        return []

    print(f"\nEvaluating {model_name} from: {weights_path}")
    start_time = time.time()
    val_results = _execute_validation(weights_path, args)
    if val_results is None:
        return []

    total_time = time.time() - start_time
    print(f"Finished evaluating {model_name} in {total_time:.2f}s.")

    if not hasattr(val_results, "box"):
        print(f"Warning: No box metrics found in validation results for {model_name}.")
        return []

    # Get class names dictionary from validation results
    class_names = val_results.names
    # results.box.all_ap is a numpy array of shape (nc, 10)
    all_ap = val_results.box.all_ap

    if all_ap is None or len(all_ap) == 0:
        print(f"Warning: Empty Average Precision (all_ap) array for {model_name}.")
        return []

    return _format_results(model_name, class_names, all_ap)


def _plot_overall_comparison(
    df: pd.DataFrame,
    ap_cols: List[str],
    iou_vals: List[float],
    output_dir: str | None,
) -> None:
    """Helper to plot the overall mAP comparison across models."""
    plt.figure(figsize=(10, 6))
    overall_df = df[df["Class"] == "all"]

    for _, row in overall_df.iterrows():
        y_vals = [row[col] for col in ap_cols]
        plt.plot(
            iou_vals,
            y_vals,
            marker="o",
            linewidth=2.5,
            label=f"{row['Model']} (mAP50-95: {row['mAP50-95']:.4f})",
        )

    plt.xlabel("IoU Threshold", fontsize=12)
    plt.ylabel("mAP", fontsize=12)
    plt.title(
        "Overall mAP @ IoU Thresholds (0.50 - 0.95)", fontsize=14, fontweight="bold"
    )
    plt.xticks(iou_vals)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    if output_dir:
        model_plot_path = os.path.join(output_dir, "map50_95_models_comparison.png")
        plt.savefig(model_plot_path, dpi=300)
        plt.close()
        print(f"Saved model comparison plot to: {model_plot_path}")
    else:
        plt.show()


def _plot_single_model_class_breakdown(
    ax: Any,
    model: str,
    model_df: pd.DataFrame,
    overall_model_df: pd.DataFrame,
    ap_cols: List[str],
    iou_vals: List[float],
) -> None:
    """Helper to plot class AP lines and overall mean for one model."""
    for _, row in model_df.iterrows():
        y_vals = [row[col] for col in ap_cols]
        ax.plot(
            iou_vals,
            y_vals,
            marker="x",
            linestyle="--",
            linewidth=1.8,
            label=f"Class: {row['Class']} (mAP50-95: {row['mAP50-95']:.4f})",
        )

    # Plot overall mean curve for reference
    if not overall_model_df.empty:
        row = overall_model_df.iloc[0]
        y_vals = [row[col] for col in ap_cols]
        ax.plot(
            iou_vals,
            y_vals,
            marker="o",
            color="black",
            linewidth=3.0,
            label=f"Overall Mean (mAP50-95: {row['mAP50-95']:.4f})",
        )


def _plot_class_breakdown(
    df: pd.DataFrame,
    ap_cols: List[str],
    iou_vals: List[float],
    output_dir: str | None,
) -> None:
    """Helper to plot class-specific AP curves for each model."""
    models = df["Model"].unique()
    num_models = len(models)

    fig, axes = plt.subplots(num_models, 1, figsize=(10, 5 * num_models), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[i, 0]
        model_df = df[(df["Model"] == model) & (df["Class"] != "all")]
        overall_model_df = df[(df["Model"] == model) & (df["Class"] == "all")]

        _plot_single_model_class_breakdown(
            ax, model, model_df, overall_model_df, ap_cols, iou_vals
        )

        ax.set_xlabel("IoU Threshold", fontsize=11)
        ax.set_ylabel("AP / mAP", fontsize=11)
        ax.set_title(
            f"Class Performance Range: {model}", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(iou_vals)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    if output_dir:
        class_plot_path = os.path.join(output_dir, "map50_95_classes_analysis.png")
        plt.savefig(class_plot_path, dpi=300)
        plt.close()
        print(f"Saved class-specific analysis plot to: {class_plot_path}")
    else:
        plt.show()


def _get_ap_columns_and_vals(df: pd.DataFrame) -> tuple:
    """Helper to filter out AP threshold columns and calculate floats."""
    ap_cols = [col for col in df.columns if col.startswith("AP@")]
    iou_vals = [int(col.split("@")[1]) / 100.0 for col in ap_cols]
    return ap_cols, iou_vals


def generate_plots(df: pd.DataFrame, output_dir: str | None = None) -> None:
    """Generate comparison plots from the detailed mAP data and save/show them.

    Args:
        df: Pandas DataFrame containing the evaluation results.
        output_dir: Directory where the plots will be saved. If None,
                    displays the plots inline using plt.show().
    """
    if df.empty:
        return

    # Ensure output directory exists if saving
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ap_cols, iou_vals = _get_ap_columns_and_vals(df)

    _plot_overall_comparison(df, ap_cols, iou_vals, output_dir)
    _plot_class_breakdown(df, ap_cols, iou_vals, output_dir)


def _print_row(row: pd.Series, ap_cols: List[str]) -> None:
    """Helper to print a single table row formatted cleanly."""
    val_str = " | ".join([f"{row[col]:<6.4f}" for col in ap_cols])
    print(
        f"{row['Model']:<25} | {row['Class']:<12} | {val_str} | {row['mAP50-95']:.4f}"
    )


def _print_rows(rows_df: pd.DataFrame, ap_cols: List[str]) -> None:
    """Helper to print multiple DataFrame rows formatted cleanly."""
    for _, row in rows_df.iterrows():
        _print_row(row, ap_cols)


def print_comparison_tables(df: pd.DataFrame) -> None:
    """Print results in clean tables printed to console.

    Args:
        df: Pandas DataFrame containing the detailed metrics.
    """
    if df.empty:
        return

    # Filter and format for display
    print("\n" + "=" * 115)
    print("YOLO MODEL DETAILED PERFORMANCE BETWEEN mAP50 AND mAP95")
    print("=" * 115)

    ap_cols = [col for col in df.columns if col.startswith("AP@")]

    # Header
    header = (
        f"{'Model':<25} | {'Class':<12} | "
        + " | ".join([f"{col:<6}" for col in ap_cols])
        + " | mAP50-95"
    )
    print(header)
    print("-" * 115)

    # Print overall rows first
    _print_rows(df[df["Class"] == "all"], ap_cols)

    print("-" * 115)

    # Print class specific rows grouped by model
    models = df["Model"].unique()
    for model in models:
        class_df = df[(df["Model"] == model) & (df["Class"] != "all")]
        _print_rows(class_df, ap_cols)
        print("-" * 115)

    print("=" * 115)


def main() -> None:
    """Run the IoU threshold range analysis pipeline."""
    args = parse_arguments()

    # Determine which models to evaluate
    if args.weights:
        model_name = os.path.basename(args.weights)
        models_to_eval = {model_name: args.weights}
    else:
        models_to_eval = DEFAULT_MODELS

    all_results = []
    for name, path in models_to_eval.items():
        metrics = run_iou_analysis(name, path, args)
        if metrics:
            all_results.extend(metrics)

    if not all_results:
        print("No models were successfully evaluated across the IoU range.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    csv_output = os.path.join(args.output_dir, "map50_95_analysis.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(csv_output, index=False)
    print(f"\nSaved detailed analysis CSV to: {csv_output}")

    # Print tabular comparison in console
    print_comparison_tables(df)

    # Generate comparison plots
    generate_plots(df, args.output_dir)


if __name__ == "__main__":
    main()
