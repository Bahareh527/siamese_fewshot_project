from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LabelledDataset(Protocol):
    targets: list[int]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]: ...


def _class_index(targets: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        grouped[int(label)].append(index)
    if len(grouped) < 2:
        raise ValueError("At least two classes are required.")
    return dict(grouped)


class ImageFolderDataset(Dataset):
    """Minimal class-folder image dataset with deterministic class ordering."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, image_size: int = 84) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image root not found: {self.root}")
        self.image_size = image_size
        class_directories = sorted(path for path in self.root.iterdir() if path.is_dir())
        self.class_to_index = {path.name: index for index, path in enumerate(class_directories)}
        self.samples: list[tuple[Path, int]] = []
        for directory in class_directories:
            label = self.class_to_index[directory.name]
            self.samples.extend(
                (path, label)
                for path in sorted(directory.iterdir())
                if path.suffix.lower() in self.extensions
            )
        if not self.samples:
            raise ValueError(f"No supported images found below {self.root}")
        self.targets = [label for _, label in self.samples]
        _class_index(self.targets)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB").resize((self.image_size, self.image_size)),
                dtype=np.float32,
            ).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        return tensor, label


class PairDataset(Dataset):
    """Generate deterministic positive and negative pairs from a labelled dataset."""

    def __init__(
        self,
        dataset: LabelledDataset,
        length: int,
        *,
        positive_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        if length <= 0 or not 0 <= positive_probability <= 1:
            raise ValueError("length must be positive and probability must lie in [0, 1].")
        self.dataset = dataset
        self.length = length
        self.positive_probability = positive_probability
        self.seed = seed
        self.by_class = _class_index(dataset.targets)
        if any(len(indices) < 2 for indices in self.by_class.values()):
            raise ValueError("Every class needs at least two examples for positive pairs.")
        self.classes = sorted(self.by_class)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + index)
        first_class = rng.choice(self.classes)
        positive = rng.random() < self.positive_probability
        if positive:
            first_index, second_index = rng.sample(self.by_class[first_class], 2)
            same_class = 1.0
        else:
            second_class = rng.choice([label for label in self.classes if label != first_class])
            first_index = rng.choice(self.by_class[first_class])
            second_index = rng.choice(self.by_class[second_class])
            same_class = 0.0
        first, _ = self.dataset[first_index]
        second, _ = self.dataset[second_index]
        return first, second, torch.tensor(same_class, dtype=torch.float32)


class TripletDataset(Dataset):
    """Generate deterministic anchor-positive-negative triplets."""

    def __init__(self, dataset: LabelledDataset, length: int, *, seed: int = 42) -> None:
        if length <= 0:
            raise ValueError("length must be positive.")
        self.dataset = dataset
        self.length = length
        self.seed = seed
        self.by_class = _class_index(dataset.targets)
        if any(len(indices) < 2 for indices in self.by_class.values()):
            raise ValueError("Every class needs at least two examples for triplets.")
        self.classes = sorted(self.by_class)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + index)
        anchor_class = rng.choice(self.classes)
        negative_class = rng.choice([label for label in self.classes if label != anchor_class])
        anchor_index, positive_index = rng.sample(self.by_class[anchor_class], 2)
        negative_index = rng.choice(self.by_class[negative_class])
        anchor, _ = self.dataset[anchor_index]
        positive, _ = self.dataset[positive_index]
        negative, _ = self.dataset[negative_index]
        return anchor, positive, negative
