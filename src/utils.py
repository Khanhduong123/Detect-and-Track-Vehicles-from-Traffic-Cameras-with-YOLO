"""Utility and configuration helper functions for YOLO training."""

import argparse
import glob
import os
from typing import Any, Dict

import yaml
from tqdm import tqdm as tqdm_bar
from ultralytics import YOLO

# Force MLflow to use sqlite database in the project directory
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Traffic-Vehicle-Detection"

try:
    import mlflow
except ImportError:
    mlflow = None

# Monkey-patch MLflow log_params to prevent crashes when resuming runs with modified parameters
if mlflow is not None:
    try:
        original_log_params = mlflow.log_params

        def safe_log_params(params: dict, *args, **kwargs):
            active_run = mlflow.active_run()
            if active_run:
                try:
                    client = mlflow.tracking.MlflowClient()
                    run_data = client.get_run(active_run.info.run_id).data
                    existing_params = run_data.params

                    # Only log parameters that are either new or have the exact same value
                    filtered_params = {}
                    for k, v in params.items():
                        if k in existing_params:
                            if str(v) == str(existing_params[k]):
                                filtered_params[k] = v
                        else:
                            filtered_params[k] = v

                    if filtered_params:
                        original_log_params(filtered_params, *args, **kwargs)
                except Exception as e:
                    print(f"Warning in patched mlflow.log_params: {e}")
                    original_log_params(params, *args, **kwargs)
            else:
                original_log_params(params, *args, **kwargs)

        mlflow.log_params = safe_log_params
    except Exception:
        pass


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
    """Parse command line arguments for YOLO training.

    Args:
        args_list: List of arguments to parse. If None, uses sys.argv[1:].

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO on joint HUTECH and UA-DETRAC dataset."
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
        help="YOLO model version to train, e.g., yolov8m, yolov9m (with or without .pt extension).",
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


def setup_mlflow_resume(run_name: str | None, experiment_name: str) -> None:
    """Check if there is an existing MLflow run with the given name and set MLFLOW_RUN_ID to resume it."""
    if not run_name or mlflow is None:
        return
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)

        # Search for runs with the matching run name in the experiment
        filter_str = f"tags.mlflow.runName = '{run_name}'"
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=filter_str,
            output_format="pandas",
        )

        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            print(
                f"Found existing MLflow run '{run_name}' with ID: {run_id}. Setting MLFLOW_RUN_ID to resume logging."
            )
            os.environ["MLFLOW_RUN_ID"] = run_id
        else:
            print(
                f"No existing MLflow run found for '{run_name}' in experiment '{experiment_name}'. A new run will be started."
            )
    except Exception as e:
        print(f"Warning: Failed to setup MLflow resume for run '{run_name}': {e}")


def _find_resume_checkpoint(model_name: str, project: str, name: str) -> str:
    """Find the best last.pt checkpoint path for resuming training."""
    if os.path.exists(model_name) and model_name.endswith("last.pt"):
        return model_name

    candidates = [
        os.path.join(project, name, "weights", "last.pt"),
        os.path.join("runs/detect/runs/detect", name, "weights", "last.pt"),
        os.path.join(name, "weights", "last.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    search_pattern = os.path.join("**", name, "weights", "last.pt")
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        # Sort matches by modification time descending to select the latest checkpoint
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    return ""


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
        checkpoint = _find_resume_checkpoint(model_name, project, name)
        if checkpoint and os.path.exists(checkpoint):
            print(f"Resuming training from checkpoint: {checkpoint}")
            return YOLO(checkpoint)

        raise FileNotFoundError(
            f"Error: Resume requested, but checkpoint file 'last.pt' for run '{name}' "
            f"could not be found in {project} or workspace."
        )
    else:
        # Check if the model exists locally or inside the weights/ folder
        if os.path.exists(model_name):
            print(f"Loading pretrained model from path: {model_name}")
            return YOLO(model_name)

        weights_path = os.path.join("weights", model_name)
        if os.path.exists(weights_path):
            print(f"Loading pretrained model from local weights: {weights_path}")
            return YOLO(weights_path)

        print(f"Loading pretrained model: {model_name}")
        return YOLO(model_name)


def get_model_name(training_kwargs: Dict[str, Any]) -> str:
    """Extract and validate the model name from training kwargs."""
    model_raw = training_kwargs.pop("model", "yolov8m.pt")
    if not isinstance(model_raw, str) or not model_raw.strip():
        model_name = "yolov8m.pt"
    else:
        model_name = model_raw.strip()

    _, ext = os.path.splitext(model_name)
    if not ext:
        model_name += ".pt"
    return model_name


def evaluate_model(model: YOLO) -> None:
    """Evaluate model on the test split and report metrics."""
    print("Evaluating model on the test set...")
    val_results = model.val(split="test")
    print("Validation on test split completed.")
    if val_results is not None and hasattr(val_results, "results_dict"):
        print("Test Metrics:")
        for metric, val in val_results.results_dict.items():
            print(f"  {metric}: {val:.4f}")


def register_tqdm_callbacks(model: YOLO) -> None:
    """Register custom tqdm progress bar callbacks on the YOLO model."""

    def on_train_epoch_start(trainer):
        # get total batches in the training dataloader
        total_batches = (
            len(trainer.train_loader) if hasattr(trainer, "train_loader") else 1
        )
        trainer.custom_pbar = tqdm_bar(
            total=total_batches,
            desc=f"Epoch {trainer.epoch + 1}/{trainer.epochs}",
            leave=True,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

    def on_train_batch_end(trainer):
        if hasattr(trainer, "custom_pbar"):
            # Update progress bar with loss values
            if hasattr(trainer, "tloss") and trainer.tloss is not None:
                metrics_dict = trainer.label_loss_items(trainer.tloss, prefix="train")
                postfix = {k: f"{v:.4f}" for k, v in metrics_dict.items()}
                trainer.custom_pbar.set_postfix(postfix)
            trainer.custom_pbar.update(1)

    def on_train_epoch_end(trainer):
        if hasattr(trainer, "custom_pbar"):
            trainer.custom_pbar.close()

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
