import csv
import json
import torch
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from src.attacks.backdoor.poisoning import poison_sample

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Find all class folders in a directory."""
    classes = sorted(
        [item.name for item in Path(directory).iterdir() if item.is_dir()]
    )

    if len(classes) == 0:
        raise FileNotFoundError(f"No class folders found in '{directory}'")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ImageDataset(Dataset):
    """Dataset backed by a manifest, with optional label overrides."""

    def __init__(
        self,
        targ_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        subset_ratio: float = 1.0,
        seed: Optional[int] = None,
        *,
        manifest_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        in_subset: Optional[int] = None,
        label_overrides: Optional[Mapping[str, int]] = None,
    ):
        if manifest_path is None:
            if targ_dir is None:
                raise ValueError("manifest_path or targ_dir is required.")
            self.root = Path(targ_dir)
            self.classes, self.class_to_idx = find_classes(targ_dir)
            self.samples = []
            for class_name in self.classes:
                for path in sorted(
                    path for extension in IMAGE_EXTENSIONS
                    for path in (self.root / class_name).glob(extension)
                ):
                    self.samples.append({
                        "sample_id": f"{class_name}/{path.name}",
                        "path": path,
                        "label": self.class_to_idx[class_name],
                    })
            if subset_ratio != 1.0:
                raise ValueError(
                    "Directory-based loading does not support subsetting; "
                    "use a prepared manifest."
                )
        else:
            if subset_ratio != 1.0 or seed is not None:
                raise ValueError(
                    "Subset selection belongs in the manifest, not the Dataset."
                )
            manifest = Path(manifest_path)
            self.root = Path(dataset_root) if dataset_root else manifest.parent
            with manifest.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            if not rows:
                raise ValueError(f"Manifest is empty: {manifest}")
            mapping_file = (
                Path(class_mapping_path)
                if class_mapping_path
                else manifest.parent / "class_to_idx.json"
            )
            with mapping_file.open(encoding="utf-8") as file:
                self.class_to_idx = {
                    str(name): int(label) for name, label in json.load(file).items()
                }
            self.classes = [
                name for name, _ in sorted(
                    self.class_to_idx.items(), key=lambda item: item[1]
                )
            ]
            if in_subset is not None:
                rows = [row for row in rows if int(row.get("in_subset", 0)) == in_subset]
            self.samples = []
            for row in rows:
                if row["class_name"] not in self.class_to_idx:
                    raise ValueError(f"Unknown class in manifest: {row['class_name']}")
                path = self.root / row["rel_path"]
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest image not found: {path}")
                self.samples.append({
                    "sample_id": row["sample_id"],
                    "path": path,
                    "label": int(row["label"]),
                })
        if not self.samples:
            raise RuntimeError("Manifest or dataset contains no images.")
        self.transform = transform
        self.label_overrides = dict(label_overrides or {})

    def __len__(self):
        return len(self.samples)

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.samples[index]["path"]).convert("RGB")

    def __getitem__(self, index: int):
        image = self.load_image(index)
        sample_id = str(self.samples[index]["sample_id"])
        label = self.label_overrides.get(sample_id, int(self.samples[index]["label"]))

        if self.transform:
            image = self.transform(image)

        return image, label, sample_id


class BackdoorDataset(ImageDataset):
    def __init__(
        self,
        targ_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        subset_ratio: float = 1.0,
        seed: Optional[int] = None,
        *,
        manifest_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        in_subset: Optional[int] = None,
        label_overrides: Optional[Mapping[str, int]] = None,
        trigger_config: Mapping,
    ):
        super().__init__(
            targ_dir=targ_dir,
            transform=transform,
            subset_ratio=subset_ratio,
            seed=seed,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            class_mapping_path=class_mapping_path,
            in_subset=in_subset,
            label_overrides=label_overrides,
        )

        if manifest_path is None:
            raise ValueError("BackdoorDataset requires a poisoned manifest.")

        self.trigger_config = trigger_config

        manifest_columns = set(self.samples[0].keys())
        if "is_poisoned" not in manifest_columns:
            raise ValueError("Manifest must contain is_poisoned column.")

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = self.load_image(index)
        sample_id = str(sample["sample_id"])

        is_poisoned = sample["is_poisoned"]

        image = poison_sample(
            image=image,
            is_poisoned=is_poisoned,
            trigger_config=self.trigger_config,
        )

        label = self.label_overrides.get(
            sample_id,
            int(sample["label"])
        )

        if self.transform:
            image = self.transform(image)

        return image, label, sample_id



def create_transform(size: Tuple[int, int],
                     num_magnitude_bins: int = 31,
                     train: bool=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transform

def create_dataloader(dataset,
                      batch_size,
                      num_workers,
                      shuffle=True,
                      generator: Optional[torch.Generator] = None,
                      worker_init_fn=None):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return dataloader