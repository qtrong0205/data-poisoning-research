from pathlib import Path
from shutil import copy2, rmtree
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    random_state: int = 42,
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

    # Remove old splits to avoid stale images.
    if output_dir.exists():
        rmtree(output_dir)

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
