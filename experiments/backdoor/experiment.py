import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from src.data.loading.data_setup import (
    BackdoorDataset,
    ImageDataset,
    create_dataloader,
    create_transform,
)
from src.evaluation.attacks.backdoor.eval import evaluate_model
from src.models.baseline.engine import train
from src.models.baseline.model_builder import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/attacks/backdoor/backdoor.yaml"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs/dataset/dataset.yaml"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs/baseline/baseline.yaml"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Invalid backdoor YAML configuration.")
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _experiment_variants(config: dict[str, Any], experiment_id: str):
    experiments = config["experiments"]
    if experiment_id not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    experiment = experiments[experiment_id]
    trigger = config["attack"]["trigger"]
    rates = experiment.get("poisoning_rates")
    if not rates:
        raise ValueError(f"Poisoning rates are not configured for {experiment_id}.")
    seeds = experiment.get("seeds", [experiment.get("seed", 42)])
    sizes = experiment.get("trigger_sizes_px", [experiment.get("trigger_size_px", "default")])
    positions = experiment.get("trigger_positions", [experiment.get("trigger_position", "default")])
    for rate in rates:
        for size in sizes:
            for position in positions:
                for seed in seeds:
                    yield rate, (
                        trigger["default_size_px"] if size == "default" else size
                    ), (
                        trigger["default_position"] if position == "default" else position
                    ), seed


def _run_dir(experiment: dict[str, Any], rate: float, size: int, position: str, seed: int) -> Path:
    output = _resolve(experiment["output_dir"])
    return output / f"rate_{rate:.2f}" / f"size_{size}" / position / f"seed_{seed}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_one(experiment_id: str, rate: float, trigger_size: int,
            trigger_position: str, seed: int) -> dict[str, Any]:
    config = load_config()
    dataset_config = _load_yaml(DATASET_CONFIG_PATH)
    baseline_config = _load_yaml(BASELINE_CONFIG_PATH)
    experiment = config["experiments"][experiment_id]
    data_config = config["data"]
    attack = config["attack"]
    target_label = int(attack["target"]["label"])
    run_dir = _run_dir(experiment, rate, trigger_size, trigger_position, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    poisoned_manifest = (
        _resolve(data_config["poisoned_manifest_root"])
        / experiment_id / f"rate_{rate:.2f}" / "manifest_train.csv"
    )
    if not poisoned_manifest.is_file():
        raise FileNotFoundError(
            f"Poisoned training manifest not found: {poisoned_manifest}. "
            f"Run src/attacks/backdoor/prepare_data.py for {experiment_id} first."
        )
    train_metadata_path = poisoned_manifest.parent / "metadata.json"
    metadata = json.loads(train_metadata_path.read_text(encoding="utf-8")) if train_metadata_path.exists() else {}

    clean_manifest = _resolve(data_config["test_manifest"])
    mapping_path = _resolve(data_config["class_mapping"])
    split_root = clean_manifest.parent
    with mapping_path.open(encoding="utf-8") as file:
        class_to_idx = json.load(file)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    train_transform = create_transform(tuple(dataset_config["dataset"]["image_size"]), train=True)
    eval_transform = create_transform(tuple(dataset_config["dataset"]["image_size"]), train=False)
    trigger_config = {
        "size_px": trigger_size,
        "color": attack["trigger"]["color"],
        "position": trigger_position,
    }
    training = config["training"]
    train_dataset = BackdoorDataset(
        manifest_path=str(poisoned_manifest), dataset_root=str(split_root),
        class_mapping_path=str(mapping_path), transform=train_transform,
        trigger_config=trigger_config,
    )
    clean_dataset = ImageDataset(
        manifest_path=str(clean_manifest), dataset_root=str(split_root),
        class_mapping_path=str(mapping_path), transform=eval_transform,
    )

    # Build an ASR manifest from the original test manifest, preserving source labels.
    test_frame = pd.read_csv(clean_manifest)
    asr_frame = test_frame.loc[test_frame["label"] != target_label].copy()
    if asr_frame.empty:
        raise ValueError("ASR test set is empty after excluding the target class.")
    asr_frame["is_poisoned"] = True
    asr_manifest = run_dir / "asr_test_manifest.csv"
    asr_frame.to_csv(asr_manifest, index=False)
    asr_dataset = BackdoorDataset(
        manifest_path=str(asr_manifest), dataset_root=str(split_root),
        class_mapping_path=str(mapping_path), transform=eval_transform,
        trigger_config=trigger_config,
    )

    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(baseline_config["training"].get("num_workers", 0)),
        "worker_init_fn": _worker_init_fn,
    }
    train_loader = create_dataloader(train_dataset, shuffle=True, generator=generator, **loader_args)
    clean_loader = create_dataloader(clean_dataset, shuffle=False, **loader_args)
    asr_loader = create_dataloader(asr_dataset, shuffle=False, **loader_args)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=len(class_names), device=device,
        pretrained=bool(baseline_config["model"].get("pretrained", True)),
    )
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(training["learning_rate"]),
    )
    checkpoint = run_dir / "final_model.pth"
    history = train(
        model, train_loader, torch.nn.CrossEntropyLoss(), optimizer,
        int(experiment["epochs"]), device, str(checkpoint),
    )
    # Evaluate the saved checkpoint so the artifact on disk is the evaluated model.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    results = evaluate_model(
        model, clean_loader, asr_loader, device, target_label,
        class_names=class_names,
        clean_predictions_path=run_dir / "clean_predictions.csv",
        asr_predictions_path=run_dir / "asr_predictions.csv",
    )
    figure = results["clean"]["confusion_matrix"]
    figure.savefig(run_dir / "clean_confusion_matrix.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)

    run_metadata = {
        "experiment_id": experiment_id,
        "experiment_name": experiment["name"],
        "poisoning_rate": rate,
        "trigger": trigger_config,
        "seed": seed,
        "target_label": target_label,
        "train_samples": len(train_dataset),
        "clean_test_samples": len(clean_dataset),
        "asr_test_samples": len(asr_dataset),
        "training_manifest": str(poisoned_manifest),
        "training_metadata": metadata,
        "epochs": int(experiment["epochs"]),
        "training_history": history,
        "evaluation": {
            "clean": results["clean"]["metrics"],
            "backdoor": {
                "metrics": results["backdoor"]["metrics"],
                "per_class_asr": results["backdoor"]["per_class_asr"],
            },
            "artifacts": {
                "checkpoint": str(checkpoint),
                "clean_predictions": results["clean"]["predictions_path"],
                "asr_predictions": results["backdoor"]["predictions_path"],
                "clean_confusion_matrix": str(run_dir / "clean_confusion_matrix.png"),
            },
        },
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(run_metadata), file, indent=2)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({"config": config, "run": _json_safe(run_metadata)}, file, sort_keys=False)
    return run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(load_config()["experiments"]), required=True)
    args = parser.parse_args()
    config = load_config()
    experiment = config["experiments"][args.experiment]
    variants = list(_experiment_variants(config, args.experiment))
    for rate, size, position, seed in tqdm(variants, desc=args.experiment, unit="run"):
        run_one(args.experiment, rate, size, position, seed)


if __name__ == "__main__":
    main()
