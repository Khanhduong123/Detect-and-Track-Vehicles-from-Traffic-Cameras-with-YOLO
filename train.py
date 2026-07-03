"""YOLOv8m training pipeline for vehicle detection in traffic cameras."""

import argparse
import os
from typing import Any, Dict

import torch
import yaml
from ultralytics import YOLO

# Standard/Fallback default configuration parameters
FALLBACK_DEFAULTS: Dict[str, Any] = {
    "model": "yolov8m.pt",
    "data": "dataset/dataset.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "",
    "workers": 8,
    "project": "runs/detect",
    "name": "yolov8m_traffic",
    "optimizer": "AdamW",
    "lr0": 0.01,
    "box": 7.5,
    "cls": 0.5,
    "fraction": 1.0,
    "resume": False,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.4,
    "degrees": 10.0,
    "scale": 0.5,
    "fliplr": 0.5,
}


def parse_args(args_list: Any = None) -> argparse.Namespace:
    """Parse command line arguments for YOLOv8m training.

    Args:
        args_list: List of arguments to parse. If None, uses sys.argv[1:].

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv8m on joint HUTECH and UA-DETRAC dataset."
    )
    # Configuration file parameter
    parser.add_argument(
        "--cfg",
        type=str,
        default="hyp.yaml",
        help="Path to hyperparameters and training configuration YAML file.",
    )

    # Set default=None for all options to identify explicit CLI overrides
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["yolov8m.pt", "yolov9m.pt", "yolov8m", "yolov9m"],
        help="YOLO model version to train (yolov8m.pt or yolov9m.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (use -1 for auto-batching).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name / output directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=[
            "SGD",
            "Adam",
            "Adamax",
            "AdamW",
            "Nadam",
            "RAdam",
            "RMSProp",
            "auto",
        ],
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=None,
        help="Box loss gain weight.",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=None,
        help="Class loss gain weight.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of dataset to train on.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )

    # CLI arguments for augmentations
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--copy_paste", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)

    return parser.parse_args(args_list)


def load_yaml_config(cfg_path: str) -> Dict[str, Any]:
    """Load config from YAML if path exists and is valid.

    Args:
        cfg_path: The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration parameters.
    """
    if not (cfg_path and os.path.exists(cfg_path)):
        return {}
    try:
        with open(cfg_path, "r") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                print(f"Loaded configuration parameters from {cfg_path}")
                return loaded
    except (OSError, yaml.YAMLError) as e:
        print(f"Warning: Failed to load config from {cfg_path}: {e}")
    return {}


def merge_configs(args: argparse.Namespace, cfg_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI overrides, YAML configuration, and fallback defaults.

    Args:
        args: Parsed command-line arguments.
        cfg_data: Loaded YAML configuration settings.

    Returns:
        Dict[str, Any]: The final consolidated parameters.
    """
    training_kwargs: Dict[str, Any] = {}
    for key, default_val in FALLBACK_DEFAULTS.items():
        cli_val = getattr(args, key, None)
        if key == "resume":
            training_kwargs[key] = args.resume or cfg_data.get("resume", False)
        elif cli_val is not None:
            training_kwargs[key] = cli_val
        else:
            training_kwargs[key] = cfg_data.get(key, default_val)
    return training_kwargs


def load_yolo_model(model_name: str, resume: bool, project: str, name: str) -> YOLO:
    """Load pretrained model or last checkpoint to resume training.

    Args:
        model_name: Name/path of the YOLO model weight file.
        resume: Whether to resume training.
        project: Directory location for training output.
        name: Sub-folder name of the experiment.

    Returns:
        YOLO: Loaded model instance.
    """
    if resume:
        checkpoint = os.path.join(project, name, "weights", "last.pt")
        if os.path.exists(checkpoint):
            print(f"Resuming training from checkpoint: {checkpoint}")
            return YOLO(checkpoint)
        print(
            f"Error: Last checkpoint '{checkpoint}' not found. Loading base pretrained model."
        )
    else:
        print(f"Loading pretrained model: {model_name}")
    return YOLO(model_name)


def run_training(args: argparse.Namespace) -> None:
    """Train YOLO with the specified configurations."""
    cfg_data = load_yaml_config(args.cfg)
    training_kwargs = merge_configs(args, cfg_data)

    # Handle device default
    if not training_kwargs["device"]:
        training_kwargs["device"] = "0" if torch.cuda.is_available() else "cpu"

    print("=== YOLO Training Configuration ===")
    for key, val in training_kwargs.items():
        print(f"  {key:<15}: {val}")
    print("=======================================")

    model_name = training_kwargs.pop("model", "yolov8m.pt")
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    model = load_yolo_model(
        model_name,
        training_kwargs["resume"],
        training_kwargs["project"],
        training_kwargs["name"],
    )

    print("Starting training...")
    model.train(**training_kwargs)
    print("Training completed.")

    # Run validation on the test set to report final test performance
    print("Evaluating model on the test set...")
    val_results = model.val(split="test")
    print("Validation on test split completed.")
    if val_results is not None and hasattr(val_results, "results_dict"):
        print("Test Metrics:")
        for metric, val in val_results.results_dict.items():
            print(f"  {metric}: {val:.4f}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
