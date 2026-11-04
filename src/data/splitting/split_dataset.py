from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    random_state: int = None,
) -> None:
    """
    Split an image classification dataset into train and test sets.

    Source structure:
        source_dir/
            class_1/
            class_2/
            ...

    Output structure:
        output_dir/
            train/
            test/
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    classes = [folder for folder in source_dir.iterdir() if folder.is_dir()]

    for cls in classes:
        images = [
            img for img in cls.iterdir()
            if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
        ]

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