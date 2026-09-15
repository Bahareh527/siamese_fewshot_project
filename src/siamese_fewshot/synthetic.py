from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class PatternDataset(Dataset):
    """Small synthetic image dataset for software verification, not benchmarking."""

    def __init__(
        self,
        class_count: int = 6,
        samples_per_class: int = 20,
        image_size: int = 32,
        seed: int = 42,
    ) -> None:
        if class_count < 2 or samples_per_class < 2 or image_size < 16:
            raise ValueError("Use at least 2 classes, 2 samples per class, and 16-pixel images.")
        generator = torch.Generator().manual_seed(seed)
        self.images: list[torch.Tensor] = []
        self.targets: list[int] = []
        coordinates = torch.linspace(-1, 1, image_size)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        for label in range(class_count):
            angle = label * math.pi / class_count
            direction = xx * math.cos(angle) + yy * math.sin(angle)
            frequency = 2 + label % 3
            pattern = (torch.sin(frequency * math.pi * direction) > 0).float()
            color = torch.zeros(3, 1, 1)
            color[label % 3] = 0.8
            base = 0.1 + pattern.unsqueeze(0) * color + (1 - pattern).unsqueeze(0) * 0.15
            for _ in range(samples_per_class):
                noise = torch.randn(3, image_size, image_size, generator=generator) * 0.08
                self.images.append((base + noise).clamp(0, 1))
                self.targets.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]
