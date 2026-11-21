
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.attacks.Label_flipping.poisoning import flip_labels_random, flip_labels_targeted


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = "configs/attacks/label_flipping/label_flipping.yaml"


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def resolve_path(path):
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_inputs(config):
    data_config = config["data"]

    train_path = resolve_path(data_config["train_manifest"])
    mapping_path = resolve_path(data_config["class_mapping"])

    for path in [train_path, mapping_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    train_df = pd.read_csv(train_path)

    with mapping_path.open(encoding="utf-8") as file:
        class_to_idx = json.load(file)

    if "label" not in train_df.columns:
        raise ValueError("Train manifest không có cột 'label'")

    return train_df, class_to_idx


def get_experiment_settings(config, experiment_id):
    if experiment_id not in config["experiments"]:
        available = list(config["experiments"].keys())
        raise ValueError(
            f"Không tìm thấy experiment '{experiment_id}'. "
            f"Các experiment hiện có: {available}"
        )

    attack_config = config["attack"]
    experiment_config = config["experiments"][experiment_id]

    # Ưu tiên cấu hình riêng của experiment.
    # Nếu không có thì sử dụng cấu hình attack toàn cục.
    strategy = experiment_config.get(
        "strategy",
        attack_config.get("strategy", "random")
    )

    denominator = experiment_config.get(
        "denominator",
        attack_config.get(
            "denominator",
            "all" if strategy == "random" else "source_class"
        )
    )

    source_class = experiment_config.get(
        "source_class",
        attack_config.get("source_class")
    )

    target_class = experiment_config.get(
        "target_class",
        attack_config.get("target_class")
    )

    selection_seed = experiment_config.get(
        "selection_seed",
        attack_config.get("selection_seed", 42)
    )

    poisoning_rates = experiment_config.get("poisoning_rates", [])
    dataset_scope = experiment_config.get("dataset_scope", "full")

    if strategy not in ["random", "targeted"]:
        raise ValueError(f"Strategy không hợp lệ: {strategy}")

    if denominator not in ["all", "source_class"]:
        raise ValueError(f"Denominator không hợp lệ: {denominator}")

    if strategy == "random" and denominator != "all":
        raise ValueError("Random strategy phải sử dụng denominator='all'")

    if dataset_scope not in ["full", "subset"]:
        raise ValueError(f"Dataset scope không hợp lệ: {dataset_scope}")

    if not poisoning_rates:
        raise ValueError(
            f"{experiment_id} chưa có poisoning_rates để chuẩn bị dữ liệu"
        )

    for rate in poisoning_rates:
        if not 0 < rate <= 1:
            raise ValueError(f"Poisoning rate không hợp lệ: {rate}")

    return {
        "strategy": strategy,
        "denominator": denominator,
        "source_class": source_class,
        "target_class": target_class,
        "selection_seed": selection_seed,
        "poisoning_rates": poisoning_rates,
        "dataset_scope": dataset_scope,
    }


def validate_targeted_classes(settings, class_to_idx):
    source_class = settings["source_class"]
    target_class = settings["target_class"]

    if not source_class or not target_class:
        raise ValueError(
            "Targeted strategy yêu cầu source_class và target_class"
        )

    if source_class not in class_to_idx:
        raise ValueError(f"Không tìm thấy source_class: {source_class}")

    if target_class not in class_to_idx:
        raise ValueError(f"Không tìm thấy target_class: {target_class}")

    source_label = class_to_idx[source_class]
    target_label = class_to_idx[target_class]

    if source_label == target_label:
        raise ValueError("Source class và target class không được trùng nhau")

    return source_label, target_label


def select_scope_samples(train_df, dataset_scope):
    if dataset_scope == "full":
        return train_df.copy().reset_index(drop=True)

    if "in_subset" not in train_df.columns:
        raise ValueError("Không tìm thấy cột 'in_subset' trong train manifest")

    subset_mask = (
        train_df["in_subset"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    scoped_df = train_df.loc[subset_mask].copy()

    if scoped_df.empty:
        raise ValueError("Không có mẫu nào thuộc subset")

    return scoped_df.reset_index(drop=True)


def save_poisoned_data(poisoned_df, output_dir, metadata):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_train.csv"
    metadata_path = output_dir / "metadata.json"

    poisoned_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8"
    )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return manifest_path


def prepare_experiment(config, experiment_id):
    data_config = config["data"]
    attack_config = config["attack"]

    settings = get_experiment_settings(config, experiment_id)

    train_df, class_to_idx = load_inputs(config)

    strategy = settings["strategy"]
    denominator = settings["denominator"]
    selection_seed = settings["selection_seed"]
    poisoning_rates = settings["poisoning_rates"]
    dataset_scope = settings["dataset_scope"]

    source_label = None
    target_label = None

    if strategy == "targeted":
        source_label, target_label = validate_targeted_classes(
            settings,
            class_to_idx
        )

    scoped_df = select_scope_samples(train_df, dataset_scope)
    labels_array = scoped_df["label"].to_numpy()

    output_root = (
        Path(data_config["poisoned_manifest_root"])
        / experiment_id
    )

    print("\n" + "=" * 65)
    print(f"Preparing experiment: {experiment_id}")
    print(f"Strategy: {strategy.upper()}")
    print(f"Denominator: {denominator}")
    print(f"Dataset scope: {dataset_scope}")
    print(f"Total samples: {len(scoped_df)}")
    print(f"Selection seed: {selection_seed}")
    print(f"Poisoning rates: {poisoning_rates}")

    if strategy == "targeted":
        print(f"Source class: {settings['source_class']}")
        print(f"Target class: {settings['target_class']}")

    print("=" * 65)

    num_classes = len(class_to_idx)

    for rate in poisoning_rates:

        # Luôn tạo poisoning từ nhãn gốc.
        # Không sử dụng manifest đã bị poisoning ở rate trước.
        if strategy == "random":
            new_labels, poisoned_idx = flip_labels_random(
                labels_array,
                rate,
                num_classes,
                selection_seed
            )

        else:
            new_labels, poisoned_idx = flip_labels_targeted(
                labels_array,
                rate,
                source_label,
                target_label,
                selection_seed,
                denominator
            )

        poison_count = len(poisoned_idx)

        # Tạo manifest mới.
        poisoned_df = scoped_df.copy()

        poisoned_df["original_label"] = labels_array
        poisoned_df["label"] = new_labels
        poisoned_df["is_poisoned"] = False

        # poisoned_idx là vị trí trong mảng NumPy,
        # không phải index gốc của DataFrame.
        poisoned_df.iloc[
            poisoned_idx,
            poisoned_df.columns.get_loc("is_poisoned")
        ] = True

        metadata = {
            "experiment_id": experiment_id,
            "strategy": strategy,
            "denominator": denominator,
            "poisoning_rate": rate,
            "selection_seed": selection_seed,
            "dataset_scope": dataset_scope,
            "total_samples": len(scoped_df),
            "poison_count": poison_count,
            "source_class": settings["source_class"],
            "target_class": settings["target_class"],
        }

        rate_dir = f"rate_{rate:.2f}"

        manifest_path = save_poisoned_data(
            poisoned_df,
            output_root / rate_dir,
            metadata
        )

        print(
            f"\nRate: {rate:.1%}"
            f" | Poisoned: {poison_count}/{len(scoped_df)}"
            f" | Saved: {manifest_path}"
        )

    print("\nPreparation completed!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Label Flipping poisoned manifests"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment ID, ví dụ: E-LF1, E-LF2, E-LF3"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Đường dẫn đến label_flipping.yaml"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    prepare_experiment(
        config,
        args.experiment
    )


if __name__ == "__main__":
    main()
