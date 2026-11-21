"""Run configured Label Flipping experiments against prepared manifests."""

import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from src.data.loading.data_setup import (
    ImageDataset,
    LabelFlippingDataset,
    create_dataloader,
    create_transform,
)
from src.evaluation.attacks.label_flipping.eval import evaluate_model
from src.models.baseline.engine import train
from src.models.baseline.model_builder import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "attacks" / "Label_flipping" / "Label_flipping.yaml"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset" / "dataset.yaml"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline" / "baseline.yaml"


def load_config(
    config_path: Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as file:
        attack_config = yaml.safe_load(file)
    with DATASET_CONFIG_PATH.open(encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)
    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as file:
        baseline_config = yaml.safe_load(file)
    return attack_config, dataset_config, baseline_config


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


def _rate_name(rate: float) -> str:
    return f"rate_{rate:.2f}"


def _settings(config: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    experiments = config.get("experiments", {})
    if experiment_name not in experiments:
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. Available: {list(experiments)}"
        )
    settings = dict(config["attack"])
    settings.update(experiments[experiment_name])
    rates = settings.get("poisoning_rates", [])
    if not rates:
        raise ValueError(f"{experiment_name} has no poisoning_rates configured.")
    if settings.get("strategy") not in {"random", "targeted"}:
        raise ValueError("strategy must be 'random' or 'targeted'.")
    if settings.get("dataset_scope") not in {"full", "subset"}:
        raise ValueError("dataset_scope must be 'full' or 'subset'.")
    if any(not 0 < float(rate) <= 1 for rate in rates):
        raise ValueError("poisoning_rates must be between 0 and 1.")
    return settings


def _build_loaders(
    attack_config: dict[str, Any],
    dataset_config: dict[str, Any],
    baseline_config: dict[str, Any],
    experiment_name: str,
    rate: float,
    seed: int,
):
    data_config = attack_config["data"]
    dataset = dataset_config["dataset"]
    split_dir = PROJECT_ROOT / dataset["split_dir"]
    mapping = PROJECT_ROOT / data_config["class_mapping"]
    test_manifest = PROJECT_ROOT / data_config["test_manifest"]
    poisoned_manifest = (
        PROJECT_ROOT / data_config["poisoned_manifest_root"]
        / experiment_name / _rate_name(rate) / "manifest_train.csv"
    )
    required = (test_manifest, mapping, poisoned_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required manifests are missing: "
            + ", ".join(missing)
            + ". Run prepare_data.py for this experiment first."
        )

    image_size = tuple(dataset["image_size"])
    train_dataset = LabelFlippingDataset(
        manifest_path=str(poisoned_manifest),
        dataset_root=str(split_dir),
        class_mapping_path=str(mapping),
        transform=create_transform(image_size, train=True),
    )
    test_dataset = ImageDataset(
        manifest_path=str(test_manifest),
        dataset_root=str(split_dir),
        class_mapping_path=str(mapping),
        transform=create_transform(image_size, train=False),
    )
    training = baseline_config["training"]
    batch_size = training["batch_size"]
    num_workers = training["num_workers"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "generator": generator,
    }
    return (
        create_dataloader(
            train_dataset, shuffle=True, worker_init_fn=_worker_init_fn, **common
        ),
        create_dataloader(
            test_dataset, shuffle=False, worker_init_fn=_worker_init_fn, **common
        ),
        train_dataset.classes,
        train_dataset,
    )


def _completed_run(run_root: Path, checkpoint_path: Path, experiment_name: str,
                   rate: float, seed: int) -> dict[str, Any] | None:
    metrics_path = run_root / "metrics.json"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        results.get("experiment_name") != experiment_name
        or results.get("poisoning_rate") != rate
        or results.get("run_seed") != seed
    ):
        return None
    return results


def run_one(attack_config: dict[str, Any], dataset_config: dict[str, Any],
            baseline_config: dict[str, Any],
            experiment_name: str, rate: float, seed: int) -> dict[str, Any]:
    output_root = PROJECT_ROOT / "experiments" / "label_flipping"
    run_root = output_root / "results" / experiment_name / _rate_name(rate) / f"seed_{seed}"
    checkpoint_path = (
        output_root / "checkpoints" / experiment_name / _rate_name(rate)
        / f"seed_{seed}" / "final_model.pth"
    )
    completed = _completed_run(run_root, checkpoint_path, experiment_name, rate, seed)
    if completed is not None:
        print(f"{experiment_name} {_rate_name(rate)} seed {seed}: skipping completed run.")
        return completed

    set_seed(seed)
    train_loader, test_loader, class_names, train_dataset = _build_loaders(
        attack_config, dataset_config, baseline_config, experiment_name, rate, seed
    )
    settings = _settings(attack_config, experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = baseline_config["training"]
    epochs = settings.get("epochs", 1)
    model = build_model(
        num_classes=dataset_config["dataset"]["num_classes"],
        device=device,
        pretrained=baseline_config["model"]["pretrained"],
    )
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=training.get("learning_rate", 1e-3),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    training_results = train(
        model, train_loader, torch.nn.CrossEntropyLoss(), optimizer,
        epochs, device, str(checkpoint_path),
    )
    flipped_sample_ids = [
        str(sample["sample_id"])
        for sample in train_dataset.samples
        if str(sample["is_poisoned"]).strip().lower() in {"1", "true", "yes"}
    ]
    evaluation_start = perf_counter()
    evaluation = evaluate_model(
        model,
        test_loader,
        device,
        class_names=class_names,
        poisoned_train_dataloader=train_loader,
        flipped_sample_ids=flipped_sample_ids,
        predictions_path=run_root / "predictions.csv",
        return_details=True,
    )
    print(f"Evaluation completed in {perf_counter() - evaluation_start:.2f}s.", flush=True)
    figure = evaluation["clean"]["confusion_matrix"]
    figure.savefig(run_root / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    metrics = {
        "experiment_name": experiment_name,
        "poisoning_rate": float(rate),
        "run_seed": seed,
        "strategy": settings["strategy"],
        "dataset_scope": settings["dataset_scope"],
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "num_classes": len(class_names),
        "epochs": epochs,
        "training": training_results,
        "evaluation": evaluation["clean"]["metrics"],
        "attack": evaluation.get("attack", {}),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }
    (run_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        dest="experiments",
        nargs="+",
        required=True,
        help="Experiment names from the config."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    attack_config, dataset_config, baseline_config = load_config(args.config)
    for experiment_name in args.experiments:
        settings = _settings(attack_config, experiment_name)
        seeds = settings.get("seeds", [settings.get("seed", 42)])
        rates = [float(rate) for rate in settings["poisoning_rates"]]
        runs = [(rate, int(seed)) for rate in rates for seed in seeds]
        progress = tqdm(runs, desc=experiment_name, unit="run", dynamic_ncols=True)
        for rate, seed in progress:
            result = run_one(
                attack_config, dataset_config, baseline_config,
                experiment_name, rate, seed,
            )
            progress.set_postfix(
                rate=f"{rate:.2f}",
                seed=seed,
                accuracy=f"{result['evaluation']['accuracy']:.4f}",
            )


if __name__ == "__main__":
    main()
