
import argparse
import copy
import json
import random
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from src.attacks.backdoor.prepare_data import main as prepare_backdoor_data
from src.evaluation.attacks.backdoor.eval import evaluate_model
from src.models.baseline.engine import train
from src.models.baseline.model_builder import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKDOOR_CONFIG_PATH = (
    PROJECT_ROOT / "configs/attacks/backdoor/backdoor.yaml"
)
DATASET_CONFIG_PATH = (
    PROJECT_ROOT / "configs/dataset/dataset.yaml"
)
BASELINE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/baseline/baseline.yaml"
)


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def load_configs() -> tuple[dict, dict, dict]:
    return (
        load_yaml(BACKDOOR_CONFIG_PATH),
        load_yaml(DATASET_CONFIG_PATH),
        load_yaml(BASELINE_CONFIG_PATH),
    )


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    return value

def format_rate(rate: float) -> str:
    return f"{float(rate):.2f}"

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def get_experiment_config(
    backdoor_config: dict,
    experiment_id: str,
) -> dict:
    experiments = backdoor_config.get("experiments", {})

    if experiment_id not in experiments:
        raise ValueError(
            f"Experiment '{experiment_id}' does not exist. "
            f"Available experiments: {list(experiments.keys())}"
        )

    config = experiments[experiment_id]

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid configuration for {experiment_id}."
        )

    return config


def validate_experiment(
    backdoor_config: dict,
    dataset_config: dict,
    baseline_config: dict,
    experiment_id: str,
) -> dict:
    experiment = get_experiment_config(
        backdoor_config,
        experiment_id,
    )

    if not experiment.get("poisoning_rates"):
        raise ValueError(
            f"{experiment_id}: 'poisoning_rates' is empty or null. "
            "Please configure the poisoning rates before running."
        )

    if experiment.get("epochs") is None or experiment["epochs"] <= 0:
        raise ValueError(
            f"{experiment_id}: epochs must be a positive integer."
        )

    if experiment.get("dataset_scope") not in {"subset", "full"}:
        raise ValueError(
            f"{experiment_id}: dataset_scope must be 'subset' or 'full'."
        )

    subset_ratio = experiment.get("subset_ratio")

    if experiment["dataset_scope"] == "subset":
        if subset_ratio is None or not 0 < subset_ratio <= 1:
            raise ValueError(
                f"{experiment_id}: subset_ratio must be in (0, 1]."
            )

    if not experiment.get("seeds") and experiment.get("seed") is None:
        raise ValueError(
            f"{experiment_id}: configure either 'seed' or 'seeds'."
        )

    if not dataset_config.get("dataset", {}).get("num_classes"):
        raise ValueError("dataset.num_classes is missing.")

    if not dataset_config.get("dataset", {}).get("image_size"):
        raise ValueError("dataset.image_size is missing.")

    if baseline_config.get("model", {}).get("pretrained") is None:
        raise ValueError("baseline.yaml: model.pretrained is missing.")

    training_config = baseline_config.get("training", {})

    if not training_config.get("batch_size"):
        raise ValueError("baseline.yaml: training.batch_size is missing.")

    if training_config.get("num_workers") is None:
        raise ValueError("baseline.yaml: training.num_workers is missing.")

    if not backdoor_config.get("attack", {}).get("target", {}).get("label") is not None:
        raise ValueError("backdoor.yaml: attack.target.label is missing.")

    return experiment


def get_trigger_sizes(
    experiment: dict,
    trigger_config: dict,
) -> list[int]:
    sizes = experiment.get("trigger_sizes_px")

    if sizes is not None:
        return sizes

    size = experiment.get("trigger_size_px", "default")

    if size == "default":
        size = trigger_config["default_size_px"]

    return [int(size)]


def get_trigger_positions(
    experiment: dict,
    trigger_config: dict,
) -> list[str]:
    positions = experiment.get("trigger_positions")

    if positions is not None:
        return positions

    position = experiment.get("trigger_position", "default")

    if position == "default":
        position = trigger_config["default_position"]

    return [str(position)]


def get_experiment_variants(
    experiment: dict,
    trigger_config: dict,
) -> list[dict]:
    rates = experiment["poisoning_rates"]
    sizes = get_trigger_sizes(experiment, trigger_config)
    positions = get_trigger_positions(experiment, trigger_config)

    seeds = experiment.get("seeds")

    if seeds is None:
        seeds = [experiment["seed"]]

    if not rates:
        raise ValueError("No poisoning rates configured.")

    if not sizes or not positions or not seeds:
        raise ValueError("Experiment variants cannot be empty.")

    variants = []

    for rate, size, position, seed in product(
        rates,
        sizes,
        positions,
        seeds,
    ):
        if not 0 < float(rate) <= 1:
            raise ValueError(
                f"Invalid poisoning rate: {rate}. Expected a value in (0, 1]."
            )

        if int(size) <= 0:
            raise ValueError(f"Invalid trigger size: {size}")

        variants.append({
            "poisoning_rate": float(rate),
            "trigger_size_px": int(size),
            "trigger_position": str(position),
            "seed": int(seed),
        })

    return variants


def get_variant_relative_path(variant: dict) -> Path:
    rate = format_rate(variant["poisoning_rate"])
    size = variant["trigger_size_px"]
    position = variant["trigger_position"]
    seed = variant["seed"]

    return (
        Path(f"rate_{rate}")
        / f"size_{size}"
        / position
        / f"seed_{seed}"
    )


def get_run_paths(
    experiment: dict,
    variant: dict,
) -> dict[str, Path]:
    output_root = resolve_path(experiment["output_dir"])
    relative_path = get_variant_relative_path(variant)

    return {
        "results": output_root / "results" / relative_path,
        "checkpoints": output_root / "checkpoints" / relative_path,
        "logs": output_root / "logs" / relative_path,
    }


def is_completed_run(
    experiment_id: str,
    experiment: dict,
    variant: dict,
) -> bool:
    paths = get_run_paths(experiment, variant)

    metrics_path = paths["results"] / "metrics.json"
    checkpoint_path = paths["checkpoints"] / "final_model.pth"

    if not metrics_path.is_file() or not checkpoint_path.is_file():
        return False

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        metadata = metrics.get("run_metadata", {})

        expected_values = {
            "experiment_id": experiment_id,
            "poisoning_rate": variant["poisoning_rate"],
            "trigger_size_px": variant["trigger_size_px"],
            "trigger_position": variant["trigger_position"],
            "run_seed": variant["seed"],
            "epochs": experiment["epochs"],
        }

        for key, expected in expected_values.items():
            if metadata.get(key) != expected:
                return False

    except (OSError, json.JSONDecodeError, TypeError):
        return False

    return True

def ensure_poisoned_manifests(
    backdoor_config: dict,
    experiment_id: str,
    experiment: dict,
) -> None:
    data_config = backdoor_config["data"]

    manifest_root = resolve_path(
        data_config["poisoned_manifest_root"]
    )

    experiment_manifest_root = manifest_root / experiment_id

    required_rates = experiment["poisoning_rates"]

    missing_rates = []

    for rate in required_rates:
        rate_dir = experiment_manifest_root / f"rate_{float(rate):g}"

        manifest_path = rate_dir / "manifest_train.csv"
        metadata_path = rate_dir / "metadata.json"

        if not manifest_path.is_file() or not metadata_path.is_file():
            missing_rates.append(rate)

    if not missing_rates:
        print(
            f"[{experiment_id}] Poisoned manifests already exist. "
            "Skipping preparation.",
            flush=True,
        )
        return

    print(
        f"[{experiment_id}] Preparing poisoned manifests...",
        flush=True,
    )

    # prepare_data.py generates manifests for all configured rates
    # of the selected experiment.
    prepare_backdoor_data(
        experiment_id=experiment_id,
        config_path=BACKDOOR_CONFIG_PATH,
    )

    # Verify the required outputs after preparation.
    for rate in required_rates:
        rate_dir = experiment_manifest_root / f"rate_{format_rate(rate)}"

        manifest_path = rate_dir / "manifest_train.csv"
        metadata_path = rate_dir / "metadata.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Poisoned manifest was not created: {manifest_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Manifest metadata was not created: {metadata_path}"
            )

    print(
        f"[{experiment_id}] Poisoned manifests are ready.",
        flush=True,
    )


def load_class_mapping(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return {
            str(name): int(label)
            for name, label in json.load(file).items()
        }


def build_dataloaders(
    backdoor_config: dict,
    dataset_config: dict,
    baseline_config: dict,
    experiment: dict,
    variant: dict,
    device: torch.device,
) -> tuple:
    data_config = backdoor_config["data"]
    attack_config = backdoor_config["attack"]
    dataset_settings = dataset_config["dataset"]
    training_settings = baseline_config["training"]

    rate = variant["poisoning_rate"]

    train_manifest_path = (
        resolve_path(data_config["poisoned_manifest_root"])
        / experiment.get("id", "")
        / f"rate_{format_rate(rate)}"
        / "manifest_train.csv"
    )

    # The experiment ID is assigned by the caller.
    if not train_manifest_path.is_file():
        raise FileNotFoundError(
            f"Poisoned training manifest not found: {train_manifest_path}"
        )

    clean_test_manifest_path = resolve_path(
        data_config["test_manifest"]
    )

    class_mapping_path = resolve_path(
        data_config["class_mapping"]
    )

    dataset_root = resolve_path(
        dataset_settings["split_dir"]
    )

    class_to_idx = load_class_mapping(class_mapping_path)

    target_label = int(attack_config["target"]["label"])

    if target_label not in class_to_idx.values():
        raise ValueError(
            f"Target label {target_label} is not present in class mapping."
        )

    class_names = [
        name
        for name, _ in sorted(
            class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]

    if target_label >= len(class_names):
        raise ValueError("Target label is outside the class mapping.")

    image_size = tuple(dataset_settings["image_size"])

    train_transform = create_transform(
        size=image_size,
        train=True,
    )

    test_transform = create_transform(
        size=image_size,
        train=False,
    )

    trigger_config = copy.deepcopy(attack_config["trigger"])
    trigger_config["size_px"] = variant["trigger_size_px"]
    trigger_config["position"] = variant["trigger_position"]

    train_dataset = BackdoorDataset(
        manifest_path=str(train_manifest_path),
        dataset_root=str(dataset_root),
        class_mapping_path=str(class_mapping_path),
        transform=train_transform,
        trigger_config=trigger_config,
    )

    clean_test_dataset = ImageDataset(
        manifest_path=str(clean_test_manifest_path),
        dataset_root=str(dataset_root),
        class_mapping_path=str(class_mapping_path),
        transform=test_transform,
    )

    # ASR evaluation:
    # Exclude samples that already belong to the target class.
    clean_test_df = pd.read_csv(clean_test_manifest_path)

    asr_test_df = clean_test_df[
        clean_test_df["label"].astype(int) != target_label
    ].copy()

    if asr_test_df.empty:
        raise ValueError(
            "ASR test manifest is empty after excluding target-class samples."
        )

    asr_test_df["is_poisoned"] = True

    paths = get_run_paths(experiment, variant)
    results_dir = paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    asr_manifest_path = results_dir / "asr_test_manifest.csv"
    asr_test_df.to_csv(
        asr_manifest_path,
        index=False,
    )

    asr_test_dataset = BackdoorDataset(
        manifest_path=str(asr_manifest_path),
        dataset_root=str(dataset_root),
        class_mapping_path=str(class_mapping_path),
        transform=test_transform,
        trigger_config=trigger_config,
    )

    generator = torch.Generator()
    generator.manual_seed(variant["seed"])

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=training_settings["batch_size"],
        num_workers=training_settings["num_workers"],
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    clean_test_loader = create_dataloader(
        dataset=clean_test_dataset,
        batch_size=training_settings["batch_size"],
        num_workers=training_settings["num_workers"],
        shuffle=False,
    )

    asr_test_loader = create_dataloader(
        dataset=asr_test_dataset,
        batch_size=training_settings["batch_size"],
        num_workers=training_settings["num_workers"],
        shuffle=False,
    )

    return (
        train_loader,
        clean_test_loader,
        asr_test_loader,
        class_names,
        target_label,
        trigger_config,
        str(train_manifest_path),
        str(asr_manifest_path),
    )

def run_one(
    experiment_id: str,
    backdoor_config: dict,
    dataset_config: dict,
    baseline_config: dict,
    experiment: dict,
    variant: dict,
) -> None:
    if is_completed_run(experiment_id, experiment, variant):
        print(
            f"[SKIP] {experiment_id} | "
            f"rate={variant['poisoning_rate']:g} | "
            f"size={variant['trigger_size_px']} | "
            f"position={variant['trigger_position']} | "
            f"seed={variant['seed']}",
            flush=True,
        )
        return

    seed = variant["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    paths = get_run_paths(experiment, variant)

    results_dir = paths["results"]
    checkpoints_dir = paths["checkpoints"]
    logs_dir = paths["logs"]

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "final_model.pth"
    metrics_path = results_dir / "metrics.json"
    confusion_matrix_path = results_dir / "confusion_matrix.png"
    config_snapshot_path = results_dir / "config_snapshot.yaml"
    training_history_path = results_dir / "training_history.json"
    clean_predictions_path = results_dir / "clean_predictions.csv"
    asr_predictions_path = results_dir / "asr_predictions.csv"

    print("\n" + "=" * 70)
    print(f"Experiment: {experiment_id}")
    print(f"Poisoning rate: {variant['poisoning_rate']:g}")
    print(f"Trigger size: {variant['trigger_size_px']} px")
    print(f"Trigger position: {variant['trigger_position']}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print("=" * 70, flush=True)

    (
        train_loader,
        clean_test_loader,
        asr_test_loader,
        class_names,
        target_label,
        trigger_config,
        train_manifest_path,
        asr_manifest_path,
    ) = build_dataloaders(
        backdoor_config=backdoor_config,
        dataset_config=dataset_config,
        baseline_config=baseline_config,
        experiment={
            **experiment,
            "id": experiment_id,
        },
        variant=variant,
        device=device,
    )

    # Build model using the same architecture as Baseline.
    model = build_model(
        num_classes=dataset_config["dataset"]["num_classes"],
        device=device,
        pretrained=baseline_config["model"]["pretrained"],
    )

    training_settings = baseline_config["training"]

    optimizer_name = backdoor_config["training"].get(
        "optimizer",
        training_settings.get("optimizer", "Adam"),
    )

    learning_rate = backdoor_config["training"].get(
        "learning_rate",
        training_settings["learning_rate"],
    )

    if optimizer_name.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            "Currently only Adam is configured."
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
    )

    loss_name = backdoor_config["training"].get(
        "loss",
        training_settings.get("loss", "CrossEntropyLoss"),
    )

    if loss_name != "CrossEntropyLoss":
        raise ValueError(
            f"Unsupported loss function: {loss_name}."
        )

    loss_fn = torch.nn.CrossEntropyLoss()

    print("\nStarting training...", flush=True)

    history = train(
        model=model,
        train_dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=int(experiment["epochs"]),
        device=device,
        checkpoint_path=str(checkpoint_path),
    )

    save_json(
        history,
        training_history_path,
    )

    # Load the saved checkpoint before evaluation.
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.to(device)

    print("\nStarting clean and backdoor evaluation...", flush=True)

    evaluation = evaluate_model(
        model=model,
        clean_dataloader=clean_test_loader,
        asr_dataloader=asr_test_loader,
        device=device,
        target_label=target_label,
        class_names=class_names,
        clean_predictions_path=clean_predictions_path,
        asr_predictions_path=asr_predictions_path,
        return_details=True,
    )

    clean_result = evaluation["clean"]
    backdoor_result = evaluation["backdoor"]

    # Save clean confusion matrix.
    figure = clean_result["confusion_matrix"]
    figure.savefig(
        confusion_matrix_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


    run_metadata = {
        "experiment_id": experiment_id,
        "experiment_name": experiment.get("name"),
        "mode": experiment.get("mode"),
        "poisoning_rate": variant["poisoning_rate"],
        "trigger_size_px": variant["trigger_size_px"],
        "trigger_position": variant["trigger_position"],
        "run_seed": seed,
        "epochs": int(experiment["epochs"]),
        "target_label": target_label,
        "target_class": backdoor_config["attack"]["target"]["class_name"],
        "device": str(device),
        "train_samples": len(train_loader.dataset),
        "clean_test_samples": len(clean_test_loader.dataset),
        "asr_test_samples": len(asr_test_loader.dataset),
        "train_manifest": train_manifest_path,
        "asr_manifest": asr_manifest_path,
        "checkpoint": str(checkpoint_path),
    }

    metrics = {
        "run_metadata": run_metadata,
        "trigger_config": trigger_config,
        "clean": {
            "metrics": clean_result["metrics"],
            "confusion_matrix": str(confusion_matrix_path),
            "predictions": str(clean_predictions_path),
        },
        "backdoor": {
            "metrics": backdoor_result["metrics"],
            "per_class_asr": backdoor_result["per_class_asr"],
            "target_label": backdoor_result["target_label"],
            "predictions": str(asr_predictions_path),
        },
        "training": {
            "history": history,
            "history_path": str(training_history_path),
        },
    }

    save_json(metrics, metrics_path)

    # Save the effective configuration used for this run.
    config_snapshot = {
        "backdoor": backdoor_config,
        "dataset": dataset_config,
        "baseline": baseline_config,
        "selected_experiment": experiment_id,
        "selected_variant": variant,
        "effective_trigger_config": trigger_config,
    }

    save_yaml(
        config_snapshot,
        config_snapshot_path,
    )

    # Save a short text log for this run.
    log_path = logs_dir / "run.log"

    with log_path.open("w", encoding="utf-8") as file:
        file.write(f"Experiment: {experiment_id}\n")
        file.write(f"Variant: {variant}\n")
        file.write(f"Device: {device}\n")
        file.write(f"Clean metrics: {clean_result['metrics']}\n")
        file.write(f"Backdoor metrics: {backdoor_result['metrics']}\n")
        file.write(f"Per-class ASR: {backdoor_result['per_class_asr']}\n")

    print("\nRun completed.", flush=True)
    print(f"Clean metrics: {clean_result['metrics']}")
    print(f"ASR: {backdoor_result['metrics']['asr']:.4f}")
    print(f"Results: {results_dir}", flush=True)

def run_experiment(
    experiment_id: str,
    backdoor_config: dict,
    dataset_config: dict,
    baseline_config: dict,
) -> None:
    experiment = validate_experiment(
        backdoor_config=backdoor_config,
        dataset_config=dataset_config,
        baseline_config=baseline_config,
        experiment_id=experiment_id,
    )

    trigger_config = backdoor_config["attack"]["trigger"]

    variants = get_experiment_variants(
        experiment=experiment,
        trigger_config=trigger_config,
    )

    print("\n" + "#" * 70)
    print(f"Preparing experiment: {experiment_id}")
    print(f"Experiment name: {experiment.get('name')}")
    print(f"Number of variants: {len(variants)}")
    print("#" * 70, flush=True)

    # Prepare manifests once per experiment.
    ensure_poisoned_manifests(
        backdoor_config=backdoor_config,
        experiment_id=experiment_id,
        experiment=experiment,
    )

    for variant in tqdm(
        variants,
        desc=f"Running {experiment_id}",
        unit="variant",
    ):
        run_one(
            experiment_id=experiment_id,
            backdoor_config=backdoor_config,
            dataset_config=dataset_config,
            baseline_config=baseline_config,
            experiment=experiment,
            variant=variant,
        )

    print(
        f"\nExperiment {experiment_id} finished.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Backdoor attack experiments."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--experiment",
        type=str,
        help="Experiment ID, e.g. E-BD1.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Run all configured experiments in YAML order.",
    )

    args = parser.parse_args()

    backdoor_config, dataset_config, baseline_config = load_configs()

    if args.experiment:
        run_experiment(
            experiment_id=args.experiment,
            backdoor_config=backdoor_config,
            dataset_config=dataset_config,
            baseline_config=baseline_config,
        )
        return

    if args.all:
        experiment_ids = list(
            backdoor_config.get("experiments", {}).keys()
        )

        for experiment_id in experiment_ids:
            run_experiment(
                experiment_id=experiment_id,
                backdoor_config=backdoor_config,
                dataset_config=dataset_config,
                baseline_config=baseline_config,
            )


if __name__ == "__main__":
    main()
