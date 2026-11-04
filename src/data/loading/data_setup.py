import random
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

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
    """Custom ImageFolder-style Dataset with stratified subsetting."""

    def __init__(
        self,
        targ_dir: str,
        transform=None,
        subset_ratio: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.root = Path(targ_dir)
        self.transform = transform

        if not (0 < subset_ratio <= 1):
            raise ValueError("subset_ratio must be between (0, 1].")

        self.classes, self.class_to_idx = find_classes(targ_dir)

        # Load image paths, GROUPED BY CLASS (not flattened)
        rng = random.Random(seed)
        self.paths: List[Path] = []

        for cls_name in self.classes:
            class_dir = self.root / cls_name
            class_paths: List[Path] = []
            for ext in IMAGE_EXTENSIONS:
                class_paths.extend(class_dir.glob(ext))

            if len(class_paths) == 0:
                continue  

            if subset_ratio < 1.0:
                subset_size = max(1, int(len(class_paths) * subset_ratio))
                class_paths = rng.sample(class_paths, subset_size)

            self.paths.extend(class_paths)

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in '{targ_dir}'.")

    def __len__(self):
        return len(self.paths)

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index]).convert("RGB")

    def __getitem__(self, index: int):
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label

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
                      shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader