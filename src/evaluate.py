"""Script to evaluate YOLOv8m and YOLOv9m models on the test dataset split."""

import argparse
import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO

DEFAULT_MODELS = {
    "YOLOv8m (Custom)": "runs/detect/yolov8m_traffic-4/weights/best.pt",
    "YOLOv9m (Custom)": "runs/detect/runs/detect/yolov9m_traffic_test/weights/best.pt",
    "YOLOv8m (Pretrained)": "weights/yolov8m.pt",
    "YOLOv9m (Pretrained)": "weights/yolov9m.pt",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8m and YOLOv9m models on the test set."
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
        help="Dataset split to evaluate on (val or test).",
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
    return parser.parse_args()


def run_single_evaluation(
    model_name: str,
    weights_path: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run evaluation for a single model and extract performance metrics.

    Args:
        model_name: Friendly name of the model.
        weights_path: Path to the weights .pt file.
        args: Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Extracted metrics, or empty dict if evaluation fails.
    """
    if not os.path.exists(weights_path):
        print(f"Skipping {model_name}: weights file not found at '{weights_path}'")
        return {}

    print(f"\nEvaluating {model_name} using weights from: {weights_path}")
    start_time = time.time()
    try:
        model = YOLO(weights_path)
        val_results = model.val(
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            plots=False,
        )
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {}

    total_time = time.time() - start_time
    print(f"Finished evaluating {model_name} in {total_time:.2f}s.")

    if val_results is None:
        return {}

    res_dict = val_results.results_dict
    speed_dict = val_results.speed

    return {
        "Model": model_name,
        "Weights": weights_path,
        "precision(B)": res_dict.get("metrics/precision(B)", 0.0),
        "recall(B)": res_dict.get("metrics/recall(B)", 0.0),
        "mAP50(B)": res_dict.get("metrics/mAP50(B)", 0.0),
        "mAP50-95(B)": res_dict.get("metrics/mAP50-95(B)", 0.0),
        "fitness": res_dict.get("fitness", 0.0),
        "Inference_ms": speed_dict.get("inference", 0.0),
        "Preprocess_ms": speed_dict.get("preprocess", 0.0),
        "Postprocess_ms": speed_dict.get("postprocess", 0.0),
    }


def generate_plots(df: pd.DataFrame, output_path: str) -> None:
    """Generate comparison bar plots and save as an image.

    Args:
        df: Pandas DataFrame containing the evaluation results.
        output_path: Path to save the output plot image.
    """
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    x_positions = range(len(df))
    width = 0.35

    plt.bar(
        [x - width / 2 for x in x_positions],
        df["mAP50(B)"],
        width,
        label="mAP50(B)",
        color="#1f77b4",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        df["mAP50-95(B)"],
        width,
        label="mAP50-95(B)",
        color="#ff7f0e",
    )

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title(
        f"YOLO mAP Performance Comparison ({df.iloc[0].get('Split', 'test')} split)"
    )
    plt.xticks(x_positions, df["Model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def print_comparison_table(results_list: List[Dict[str, Any]]) -> None:
    """Print results list in a neat text-aligned table format.

    Args:
        results_list: List of dictionaries containing metrics for each model.
    """
    if not results_list:
        print("No evaluation results to show.")
        return

    headers = [
        "Model",
        "precision(B)",
        "recall(B)",
        "mAP50(B)",
        "mAP50-95(B)",
        "fitness",
        "Inference (ms)",
    ]
    print("\n" + "=" * 95)
    print("YOLO MODEL TEST SPLIT EVALUATION RESULTS")
    print("=" * 95)
    print(
        f"{headers[0]:<25} | {headers[1]:<12} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<12} | {headers[5]:<10} | {headers[6]:<14}"
    )
    print("-" * 95)
    for r in results_list:
        print(
            f"{r['Model']:<25} | {r['precision(B)']:<12.4f} | {r['recall(B)']:<10.4f} | {r['mAP50(B)']:<10.4f} | {r['mAP50-95(B)']:<12.4f} | {r['fitness']:<10.4f} | {r['Inference_ms']:<14.2f}"
        )
    print("=" * 95)


def main() -> None:
    """Run the evaluation script pipeline."""
    args = parse_arguments()

    # Determine which models to evaluate
    if args.weights:
        model_name = os.path.basename(args.weights)
        models_to_eval = {model_name: args.weights}
    else:
        models_to_eval = DEFAULT_MODELS

    results = []
    for name, path in models_to_eval.items():
        metrics = run_single_evaluation(name, path, args)
        if metrics:
            results.append(metrics)

    if not results:
        print("No models were successfully evaluated.")
        return

    # Print the table comparison in console
    print_comparison_table(results)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df["Split"] = args.split

    # Save to CSV
    csv_output = "evaluation_results.csv"
    df.to_csv(csv_output, index=False)
    print(f"Saved evaluation results table to: {csv_output}")

    # Generate comparison bar chart
    plot_output = "evaluation_comparison.png"
    generate_plots(df, plot_output)


if __name__ == "__main__":
    main()
