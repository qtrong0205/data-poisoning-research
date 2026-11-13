
from pathlib import Path

import pandas as pd
import yaml
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    return config


def load_and_validate_inputs(config):
    data_config = config["data"]
    target_config = config["attack"]["target"]

    # Resolve input paths
    train_path = PROJECT_ROOT / data_config["train_manifest"]
    test_path = PROJECT_ROOT / data_config["test_manifest"]
    mapping_path = PROJECT_ROOT / data_config["class_mapping"]

    # Check file existence
    for path in [train_path, test_path, mapping_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    # Load manifests
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Load class mapping
    with mapping_path.open(encoding="utf-8") as file:
        class_to_idx = json.load(file)

    if not isinstance(class_to_idx, dict):
        raise ValueError("class_to_idx.json must contain a dictionary.")

    # Validate required columns
    required_train_cols = {
        "sample_id",
        "rel_path",
        "label",
        "class_name",
        "in_subset",
    }

    required_test_cols = {
        "sample_id",
        "rel_path",
        "label",
        "class_name",
    }

    missing_train = required_train_cols - set(train_df.columns)
    missing_test = required_test_cols - set(test_df.columns)

    if missing_train:
        raise ValueError(
            f"Train manifest missing columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Test manifest missing columns: {sorted(missing_test)}"
        )

    # Validate target class
    target_class = target_config["class_name"]
    target_label = target_config["label"]

    if target_class not in class_to_idx:
        raise ValueError(
            f"Target class not found in class mapping: {target_class}"
        )

    if class_to_idx[target_class] != target_label:
        raise ValueError(
            f"Target label mismatch: "
            f"{target_class} maps to {class_to_idx[target_class]}, "
            f"but config specifies {target_label}."
        )

    # Validate manifest labels and class names
    for name, df in [("train", train_df), ("test", test_df)]:
        if df["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate sample_id found in {name} manifest."
            )

        if df[["sample_id", "rel_path", "label", "class_name"]].isnull().any().any():
            raise ValueError(
                f"Missing required values in {name} manifest."
            )

        for class_name, label in zip(df["class_name"], df["label"]):
            if class_name not in class_to_idx:
                raise ValueError(
                    f"Unknown class '{class_name}' in {name} manifest."
                )

            if class_to_idx[class_name] != label:
                raise ValueError(
                    f"Label mismatch in {name} manifest: "
                    f"{class_name} has label {label}, "
                    f"expected {class_to_idx[class_name]}."
                )

    return train_df, test_df, class_to_idx


def select_scope_samples(train_df, dataset_scope):
    if dataset_scope not in {"subset", "full"}:
        raise ValueError(
            f"Invalid dataset scope: {dataset_scope}"
        )

    if dataset_scope == "subset":
        if "in_subset" not in train_df.columns:
            raise ValueError(
                "Column 'in_subset' not found in train manifest."
            )

        scoped_df = train_df[train_df["in_subset"] == True].copy()
    else:
        scoped_df = train_df.copy()

    if scoped_df.empty:
        raise ValueError(
            f"No samples found for dataset scope: {dataset_scope}"
        )

    return scoped_df


def get_eligible_samples(scoped_df, target_label):
    if "label" not in scoped_df.columns:
        raise ValueError("Column 'label' not found in scoped manifest.")

    eligible_df = scoped_df[
        scoped_df["label"] != target_label
    ].copy()

    if eligible_df.empty:
        raise ValueError("No eligible samples found for poisoning.")

    return eligible_df


def calculate_poison_count(eligible_count, poisoning_rate):
    if not isinstance(poisoning_rate, (int, float)):
        raise TypeError("Poisoning rate must be a number.")

    if not 0 <= poisoning_rate <= 1:
        raise ValueError("Poisoning rate must be between 0 and 1.")

    if eligible_count < 0:
        raise ValueError("Eligible count cannot be negative.")

    return round(eligible_count * poisoning_rate)


def select_poison_samples(eligible_df, poison_count, seed):
    if not isinstance(poison_count, int) or poison_count < 0:
        raise ValueError("Poison count must be a non-negative integer.")

    if poison_count > len(eligible_df):
        raise ValueError(
            f"Poison count ({poison_count}) exceeds "
            f"eligible samples ({len(eligible_df)})."
        )

    if poison_count == 0:
        return eligible_df["sample_id"].iloc[0:0].copy()

    # Generate a deterministic ordering shared across poisoning rates.
    shuffled_df = eligible_df.sample(
        frac=1,
        random_state=seed
    )

    return shuffled_df["sample_id"].head(poison_count).copy()


def build_poisoned_manifest(
    scoped_df,
    poisoned_sample_ids,
    target_label
):
    required_cols = {"sample_id", "label", "rel_path", "class_name"}

    missing_cols = required_cols - set(scoped_df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns in scoped manifest: {sorted(missing_cols)}"
        )

    poisoned_sample_ids = set(poisoned_sample_ids)

    available_sample_ids = set(scoped_df["sample_id"])

    unknown_ids = poisoned_sample_ids - available_sample_ids

    if unknown_ids:
        raise ValueError(
            f"Unknown poisoned sample IDs: {len(unknown_ids)}"
        )

    poisoned_df = scoped_df.copy()

    # Preserve original labels.
    poisoned_df["original_label"] = poisoned_df["label"]

    # Mark poisoned samples.
    poisoned_df["is_poisoned"] = poisoned_df["sample_id"].isin(
        poisoned_sample_ids
    )

    # Change labels of poisoned samples to target label.
    poisoned_df.loc[
        poisoned_df["is_poisoned"], "label"
    ] = target_label

    return poisoned_df


def save_poisoned_data(
    poisoned_df,
    output_dir,
    metadata
):
    output_dir = Path(output_dir)

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_train.csv"
    metadata_path = output_dir / "metadata.json"

    # Save poisoned manifest.
    poisoned_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8"
    )

    # Save poisoning metadata.
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    return manifest_path, metadata_path


def main(experiment_id, config_path):
    """
    Prepare poisoned manifests for a selected experiment.

    Args:
        experiment_id: Experiment ID, e.g. E-BD1.
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing generated manifest and metadata paths.
    """

    # 1. Load configuration
    config = load_config(config_path)

    # 2. Get configuration sections
    attack_config = config["attack"]
    data_config = config["data"]
    poisoning_config = attack_config["poisoning"]
    target_config = attack_config["target"]
    experiments_config = config["experiments"]

    # 3. Validate experiment ID
    if experiment_id not in experiments_config:
        raise ValueError(
            f"Experiment '{experiment_id}' not found in config."
        )

    experiment_config = experiments_config[experiment_id]

    # 4. Get experiment parameters
    poisoning_rates = experiment_config.get("poisoning_rates")

    if not poisoning_rates:
        raise ValueError(
            f"Poisoning rates are not configured for {experiment_id}."
        )

    dataset_scope = experiment_config["dataset_scope"]
    selection_seed = poisoning_config["selection_seed"]
    target_label = target_config["label"]

    # 5. Load and validate input data
    train_df, _, _ = load_and_validate_inputs(config)

    # 6. Select samples according to dataset scope
    scoped_df = select_scope_samples(
        train_df,
        dataset_scope
    )

    # 7. Get eligible samples
    eligible_df = get_eligible_samples(
        scoped_df,
        target_label
    )

    eligible_count = len(eligible_df)

    print(f"\nPreparing data for {experiment_id}")
    print(f"Dataset scope: {dataset_scope}")
    print(f"Total samples: {len(scoped_df)}")
    print(f"Eligible samples: {eligible_count}")

    # 8. Prepare output directory
    poisoned_manifest_root = Path(
        data_config["poisoned_manifest_root"]
    )

    output_root = poisoned_manifest_root / experiment_id

    results = {}

    # 9. Generate poisoned manifest for each poisoning rate
    for poisoning_rate in poisoning_rates:

        poison_count = calculate_poison_count(
            eligible_count,
            poisoning_rate
        )

        poisoned_sample_ids = select_poison_samples(
            eligible_df,
            poison_count,
            selection_seed
        )

        poisoned_df = build_poisoned_manifest(
            scoped_df,
            poisoned_sample_ids,
            target_label
        )

        # 10. Prepare metadata
        trigger_config = attack_config["trigger"]

        metadata = {
            "attack": attack_config["name"],
            "experiment_id": experiment_id,
            "dataset_scope": dataset_scope,
            "subset_ratio": experiment_config.get("subset_ratio"),
            "poisoning_rate": poisoning_rate,
            "rate_basis": poisoning_config["rate_basis"],
            "selection_seed": selection_seed,
            "target_class": target_config["class_name"],
            "target_label": target_label,
            "eligible_count": eligible_count,
            "poison_count": poison_count,
            "total_samples": len(scoped_df),
            "trigger": {
                "type": trigger_config["type"],
                "color": trigger_config["color"],
                "size_px": experiment_config.get(
                    "trigger_size_px",
                    trigger_config["default_size_px"]
                ),
                "position": experiment_config.get(
                    "trigger_position",
                    trigger_config["default_position"]
                )
            }
        }

        # 11. Save poisoned manifest and metadata
        rate_dir = f"rate_{poisoning_rate:.2f}"

        output_dir = output_root / rate_dir

        manifest_path, metadata_path = save_poisoned_data(
            poisoned_df,
            output_dir,
            metadata
        )

        results[poisoning_rate] = {
            "manifest_path": manifest_path,
            "metadata_path": metadata_path,
            "poison_count": poison_count
        }

        print(
            f"Rate: {poisoning_rate:.0%} | "
            f"Poisoned: {poison_count}/{eligible_count}"
        )
        print(f"Manifest: {manifest_path}")

    print(f"\nData preparation completed for {experiment_id}.")

    return results


if __name__ == "__main__":
    main()
