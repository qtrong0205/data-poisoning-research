"""Prepare a fixed PlantVillage split and stable experiment manifests."""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitting.split_dataset import create_manifests, split_dataset


def validate_manifests(
    split_dir: Path,
    num_classes: int,
    subset_ratio: float,
) -> None:
    """Validate existing manifests against the current split on disk."""
    train_manifest = split_dir / "manifest_train.csv"
    test_manifest = split_dir / "manifest_test.csv"
    mapping_path = split_dir / "class_to_idx.json"

    with mapping_path.open(encoding="utf-8") as file:
        class_to_idx = json.load(file)
    if len(class_to_idx) != num_classes or sorted(class_to_idx.values()) != list(
        range(num_classes)
    ):
        raise ValueError("class_to_idx.json does not contain the configured classes.")

    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    train_rows = read_rows(train_manifest)
    test_rows = read_rows(test_manifest)
    required_train = {"sample_id", "rel_path", "label", "class_name", "in_subset"}
    required_test = {"sample_id", "rel_path", "label", "class_name"}
    for path, rows, required in (
        (train_manifest, train_rows, required_train),
        (test_manifest, test_rows, required_test),
    ):
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Invalid columns or empty manifest: {path}")
        sample_ids = [row["sample_id"] for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate sample_id in manifest: {path}")
        for row in rows:
            image_path = split_dir / Path(row["rel_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image not found: {image_path}")
            if row["class_name"] not in class_to_idx:
                raise ValueError(f"Unknown class in manifest: {row['class_name']}")
            if int(row["label"]) != class_to_idx[row["class_name"]]:
                raise ValueError(f"Label mismatch in manifest: {path}")

    if len({row["class_name"] for row in train_rows}) != num_classes or len(
        {row["class_name"] for row in test_rows}
    ) != num_classes:
        raise ValueError("Manifest does not contain all configured classes.")
    for class_name in class_to_idx:
        actual_count = sum(row["class_name"] == class_name for row in train_rows)
        expected_count = len(
            [
                path
                for path in (split_dir / "train" / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if actual_count != expected_count:
            raise ValueError(f"Train sample count mismatch for class '{class_name}'.")
        expected_subset = max(1, round(actual_count * subset_ratio))
        subset_count = sum(
            row["class_name"] == class_name and row["in_subset"] == "1"
            for row in train_rows
        )
        if any(row["in_subset"] not in {"0", "1"} for row in train_rows):
            raise ValueError("Train manifest contains an invalid in_subset value.")
        if subset_count != expected_subset:
            raise ValueError(f"Invalid subset count for class '{class_name}'.")
        test_count = sum(row["class_name"] == class_name for row in test_rows)
        expected_test_count = len(
            [
                path
                for path in (split_dir / "test" / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if test_count != expected_test_count:
            raise ValueError(f"Test sample count mismatch for class '{class_name}'.")
    if set(row["sample_id"] for row in train_rows) & set(
        row["sample_id"] for row in test_rows
    ):
        raise ValueError("Train and test manifests contain duplicate sample IDs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()

    with (PROJECT_ROOT / "configs" / "dataset" / "dataset.yaml").open(
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    dataset = config["dataset"]
    split_dir = PROJECT_ROOT / dataset["split_dir"]
    raw_dir = PROJECT_ROOT / dataset["data_dir"]

    split_exists = (split_dir / "train").is_dir() and (split_dir / "test").is_dir()
    manifest_paths = [
        split_dir / "manifest_train.csv",
        split_dir / "manifest_test.csv",
        split_dir / "class_to_idx.json",
    ]
    manifests_exist = all(path.is_file() for path in manifest_paths)
    any_manifest_exists = any(path.exists() for path in manifest_paths)

    if args.force_split and any_manifest_exists and not args.force_manifest:
        raise ValueError(
            "--force-split with existing manifests also requires "
            "--force-manifest to prevent stale manifests."
        )
    if split_exists and manifests_exist and not args.force_split and not args.force_manifest:
        validate_manifests(
            split_dir,
            num_classes=dataset["num_classes"],
            subset_ratio=config["modes"]["sweep"]["subset_ratio"],
        )
        print("Dataset split and manifests are already prepared and valid.")
        return

    if args.force_split or not split_exists:
        split_dataset(
            source_dir=raw_dir,
            output_dir=split_dir,
            train_ratio=dataset["train_ratio"],
            random_state=dataset["split_seed"],
            force=args.force_split,
        )
    if any_manifest_exists and not args.force_manifest:
        raise FileExistsError(
            "Some manifest files already exist. Use --force-manifest to regenerate them."
        )
    create_manifests(
        split_dir=split_dir,
        subset_ratio=config["modes"]["sweep"]["subset_ratio"],
        subset_seed=dataset["subset_seed"],
        force=args.force_manifest,
    )
    validate_manifests(
        split_dir,
        num_classes=dataset["num_classes"],
        subset_ratio=config["modes"]["sweep"]["subset_ratio"],
    )
    print("Prepared fixed split and manifests.")


if __name__ == "__main__":
    main()
