from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(inputs) + self.skip(inputs), inplace=True)


class CompactEncoder(nn.Module):
    """Small residual image encoder that returns L2-normalized embeddings."""

    def __init__(self, embedding_dim: int = 64, base_channels: int = 16) -> None:
        super().__init__()
        if embedding_dim <= 0 or base_channels <= 0:
            raise ValueError("embedding_dim and base_channels must be positive.")
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(base_channels * 4, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return F.normalize(self.projection(features), dim=1)


class SiameseNetwork(nn.Module):
    """Shared-weight wrapper for pair, triplet, and episodic inference."""

    def __init__(self, encoder: nn.Module | None = None, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = encoder or CompactEncoder(embedding_dim=embedding_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)

    def forward_pair(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(first), self.encode(second)

    def forward_triplet(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(anchor), self.encode(positive), self.encode(negative)


def resnet18_encoder(
    embedding_dim: int = 256,
    *,
    imagenet_weights: bool = False,
    places365_checkpoint: str | Path | None = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Construct an explicit ResNet18 baseline without silently using random weights."""
    if imagenet_weights and places365_checkpoint is not None:
        raise ValueError("Choose ImageNet weights or a Places365 checkpoint, not both.")
    if not imagenet_weights and places365_checkpoint is None:
        raise ValueError("Pretrained weights must be explicitly selected for this baseline.")
    try:
        from torchvision import models
    except ImportError as error:
        raise ImportError("Install the optional 'vision' dependencies for ResNet18.") from error

    weights = models.ResNet18_Weights.DEFAULT if imagenet_weights else None
    backbone = models.resnet18(weights=weights)
    if places365_checkpoint is not None:
        source = Path(places365_checkpoint)
        if not source.is_file():
            raise FileNotFoundError(f"Places365 checkpoint not found: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        feature_state = {key: value for key, value in state.items() if not key.startswith("fc.")}
        missing, unexpected = backbone.load_state_dict(feature_state, strict=False)
        if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
            raise RuntimeError(
                f"Incompatible Places365 checkpoint; missing={missing}, unexpected={unexpected}"
            )

    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    input_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(input_features, embedding_dim),
        nn.LayerNorm(embedding_dim),
    )
    for parameter in backbone.fc.parameters():
        parameter.requires_grad = True

    class NormalizedEncoder(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.base(inputs), dim=1)

    return NormalizedEncoder(backbone)
