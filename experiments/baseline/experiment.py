"""Run the reproducible baseline experiment against fixed manifests."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.loading.data_setup import (
    ImageDataset,
    create_dataloader,
    create_transform,
)
from src.evaluation.baseline.eval import evaluate_model
from src.models.baseline.engine import train
from src.models.baseline.model_builder import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset" / "dataset.yaml"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline" / "baseline.yaml"


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    with DATASET_CONFIG_PATH.open(encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)
    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as file:
        baseline_config = yaml.safe_load(file)
    return dataset_config, baseline_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def validate_config(dataset_config: dict[str, Any], baseline_config: dict[str, Any],
                    mode: str, seed: int) -> None:
    dataset = dataset_config["dataset"]
    mode_config = dataset_config["modes"][mode]
    training = baseline_config["training"]
    if dataset["num_classes"] <= 0 or not 0 < mode_config["subset_ratio"] <= 1:
        raise ValueError("Invalid dataset class count or subset ratio.")
    if mode_config["epochs"] <= 0 or training["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if seed not in mode_config["seeds"]:
        raise ValueError(f"Seed {seed} is not configured for mode '{mode}'.")


def build_loaders(dataset_config: dict[str, Any], baseline_config: dict[str, Any],
                  mode: str, seed: int):
    dataset = dataset_config["dataset"]
    mode_config = dataset_config["modes"][mode]
    split_dir = PROJECT_ROOT / dataset["split_dir"]
    train_manifest = split_dir / "manifest_train.csv"
    test_manifest = split_dir / "manifest_test.csv"
    mapping = split_dir / "class_to_idx.json"
    missing = [path for path in (train_manifest, test_manifest, mapping) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required manifest files are missing: {missing_text}. "
            "Run 'python scripts/prepare_dataset.py' first."
        )

    image_size = tuple(dataset["image_size"])
    train_dataset = ImageDataset(
        manifest_path=str(train_manifest),
        dataset_root=str(split_dir),
        class_mapping_path=str(mapping),
        transform=create_transform(image_size, train=True),
        in_subset=1 if mode_config["use_subset"] else None,
    )
    test_dataset = ImageDataset(
        manifest_path=str(test_manifest),
        dataset_root=str(split_dir),
        class_mapping_path=str(mapping),
        transform=create_transform(image_size, train=False),
    )
    if len(train_dataset.classes) != dataset["num_classes"]:
        raise ValueError("Configured num_classes does not match class mapping.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    training = baseline_config["training"]
    common = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "generator": generator,
    }
    return (
        create_dataloader(train_dataset, shuffle=True, worker_init_fn=_worker_init_fn, **common),
        create_dataloader(test_dataset, shuffle=False, worker_init_fn=_worker_init_fn, **common),
        train_dataset.classes,
        mode_config["epochs"],
    )


def run_one(mode: str, seed: int) -> dict[str, Any]:
    dataset_config, baseline_config = load_config()
    validate_config(dataset_config, baseline_config, mode, seed)
    set_seed(seed)
    dataset = dataset_config["dataset"]
    training = baseline_config["training"]
    output_root = PROJECT_ROOT / "experiments" / "baseline"
    run_root = output_root / "results" / mode / f"seed_{seed}"
    checkpoint_path = output_root / "checkpoints" / mode / f"seed_{seed}" / "final_model.pth"
    log_path = output_root / "logs" / mode / f"seed_{seed}.log"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, class_names, epochs = build_loaders(
        dataset_config, baseline_config, mode, seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=dataset["num_classes"],
        device=device,
        pretrained=baseline_config["model"]["pretrained"],
    )
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=training["learning_rate"],
    )
    training_results = train(
        model, train_loader, torch.nn.CrossEntropyLoss(), optimizer, epochs,
        device, str(checkpoint_path),
    )
    _, figure, _, prediction_data = evaluate_model(
        model, test_loader, device, class_names=class_names,
        predictions_path=run_root / "predictions.csv",
        return_details=True,
    )
    figure.savefig(run_root / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)
    results = {
        "experiment_name": baseline_config["experiment"]["name"],
        "mode": mode,
        "run_seed": seed,
        "split_seed": dataset["split_seed"],
        "subset_seed": dataset["subset_seed"],
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "num_classes": len(class_names),
        "epochs": epochs,
        "training_config": training,
        "training": training_results,
        "evaluation": {
            metric_name: float(metric_value)
            for metric_name, metric_value in prediction_data["metrics"].items()
        },
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }
    with (run_root / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with log_path.open("w", encoding="utf-8") as file:
        evaluation = results["evaluation"]
        file.write(
            f"mode={mode} seed={seed} "
            f"accuracy={evaluation['accuracy']:.6f} "
            f"precision={evaluation['precision']:.6f} "
            f"recall={evaluation['recall']:.6f} "
            f"f1_score={evaluation['f1_score']:.6f}\n"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sweep", "confirm"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()
    dataset_config, _ = load_config()
    configured_seeds = dataset_config["modes"][args.mode]["seeds"]
    if args.all_seeds and args.seed is not None:
        parser.error("--seed and --all-seeds cannot be combined.")
    seeds = configured_seeds if args.all_seeds else [args.seed or configured_seeds[0]]
    for seed in seeds:
        result = run_one(args.mode, seed)
        evaluation = result["evaluation"]
        print(
            f"{args.mode} seed {seed}: "
            f"accuracy={evaluation['accuracy']:.4f} "
            f"precision={evaluation['precision']:.4f} "
            f"recall={evaluation['recall']:.4f} "
            f"f1_score={evaluation['f1_score']:.4f}"
        )


if __name__ == "__main__":
    main()
