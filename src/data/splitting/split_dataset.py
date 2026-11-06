import csv
import json
import random
from pathlib import Path
from shutil import copy2, rmtree
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    random_state: int = 42,
    force: bool = False,
) -> None:
    """
    Split image classification dataset into train and test.

    Source:
        source_dir/
            class_1/
            class_2/

    Output:
        output_dir/
            train/
            test/
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {source_dir}"
        )

    classes = sorted(
        folder for folder in source_dir.iterdir()
        if folder.is_dir()
    )

    if not classes:
        raise FileNotFoundError(
            f"No class folders found in {source_dir}"
        )

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Split directory already contains data: {output_dir}. "
                "Use force=True only when intentionally replacing it."
            )
        for split_name in ("train", "test"):
            split_path = output_dir / split_name
            if split_path.exists():
                rmtree(split_path)

    for cls in classes:
        images = sorted(
            img for img in cls.iterdir()
            if img.is_file()
            and img.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 2:
            raise ValueError(
                f"Class '{cls.name}' must contain at least 2 images."
            )

        train_imgs, test_imgs = train_test_split(
            images,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True,
        )

        for split_name, image_list in {
            "train": train_imgs,
            "test": test_imgs,
        }.items():

            dst_dir = output_dir / split_name / cls.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for img in image_list:
                copy2(img, dst_dir / img.name)

    print(f"Dataset successfully split into '{output_dir}'.")


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        image for image in directory.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_manifests(
    split_dir: str | Path,
    manifest_dir: str | Path | None = None,
    subset_ratio: float = 0.30,
    subset_seed: int = 42,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Create stable train/test manifests and a shared class mapping.

    Paths in the manifests are relative to ``split_dir`` so they remain
    independent of the process working directory.
    """
    if not 0 < subset_ratio <= 1:
        raise ValueError("subset_ratio must be between (0, 1].")

    split_dir = Path(split_dir)
    manifest_dir = Path(manifest_dir) if manifest_dir else split_dir
    train_dir = split_dir / "train"
    test_dir = split_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train and test directories under '{split_dir}'."
        )

    classes = sorted(
        folder.name for folder in train_dir.iterdir() if folder.is_dir()
    )
    if not classes:
        raise FileNotFoundError(f"No training class folders found in '{train_dir}'.")
    test_classes = sorted(
        folder.name for folder in test_dir.iterdir() if folder.is_dir()
    )
    if classes != test_classes:
        raise ValueError("Train and test class folders do not match.")

    class_to_idx = {name: index for index, name in enumerate(classes)}
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = manifest_dir / "manifest_train.csv"
    test_manifest = manifest_dir / "manifest_test.csv"
    mapping_path = manifest_dir / "class_to_idx.json"
    outputs = (train_manifest, test_manifest, mapping_path)
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError(
            "A manifest already exists. Use force=True to intentionally regenerate it."
        )

    rng = random.Random(subset_seed)
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()

    for split_name, split_root, rows in (
        ("train", train_dir, train_rows),
        ("test", test_dir, test_rows),
    ):
        for class_name in classes:
            images = _image_files(split_root / class_name)
            if not images:
                raise ValueError(f"Class '{class_name}' has no images in {split_name}.")
            selected = set()
            if split_name == "train":
                count = max(1, round(len(images) * subset_ratio))
                selected = {image.name for image in rng.sample(images, count)}
            for image in images:
                sample_id = f"{split_name}:{class_name}/{image.name}"
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate sample_id generated: {sample_id}")
                seen_sample_ids.add(sample_id)
                row: dict[str, object] = {
                    "sample_id": sample_id,
                    "rel_path": image.relative_to(split_dir).as_posix(),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                }
                if split_name == "train":
                    row["in_subset"] = int(image.name in selected)
                rows.append(row)

    with train_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "rel_path", "label", "class_name", "in_subset"],
        )
        writer.writeheader()
        writer.writerows(train_rows)
    with test_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "rel_path", "label", "class_name"],
        )
        writer.writeheader()
        writer.writerows(test_rows)
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(class_to_idx, file, indent=2, sort_keys=True)

    return train_manifest, test_manifest, mapping_path
