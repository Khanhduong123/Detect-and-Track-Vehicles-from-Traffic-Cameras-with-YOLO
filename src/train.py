"""YOLO training pipeline for vehicle detection in traffic cameras."""

import argparse
import os

import torch

from utils import (
    evaluate_model,
    get_model_name,
    load_yaml_config,
    load_yolo_model,
    merge_configs,
    parse_args,
    register_tqdm_callbacks,
    setup_mlflow_resume,
)


def run_training(args: argparse.Namespace) -> None:
    """Train YOLO with the specified configurations."""
    cfg_data = load_yaml_config(args.cfg)
    training_kwargs = merge_configs(args, cfg_data)

    # Handle device default
    if not training_kwargs["device"]:
        training_kwargs["device"] = "0" if torch.cuda.is_available() else "cpu"

    # Setup MLflow run ID resumption if resume is requested
    if training_kwargs.get("resume"):
        setup_mlflow_resume(
            run_name=training_kwargs.get("name"),
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", "Traffic-Vehicle-Detection"
            ),
        )

    print("=== YOLO Training Configuration ===")
    for key, val in training_kwargs.items():
        print(f"  {key:<15}: {val}")
    print("=======================================")

    model_name = get_model_name(training_kwargs)

    model = load_yolo_model(
        model_name,
        training_kwargs["resume"],
        training_kwargs["project"],
        training_kwargs["name"],
    )

    # Register custom tqdm progress bar callback for tracking training process
    register_tqdm_callbacks(model)

    print("Starting training...")
    model.train(**training_kwargs)
    print("Training completed.")

    # Run validation on the test set to report final test performance
    evaluate_model(model)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
