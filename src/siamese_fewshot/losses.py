from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss where label 1 denotes a same-class pair."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive.")
        self.margin = margin

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        same_class: torch.Tensor,
    ) -> torch.Tensor:
        labels = same_class.to(dtype=first.dtype).view(-1)
        distances = torch.linalg.vector_norm(first - second, dim=1)
        positive = labels * distances.square()
        negative = (1 - labels) * F.relu(self.margin - distances).square()
        return (positive + negative).mean()


class TripletLoss(nn.Module):
    """Squared-Euclidean triplet loss implemented from first principles."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive.")
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        positive_distance = (anchor - positive).square().sum(dim=1)
        negative_distance = (anchor - negative).square().sum(dim=1)
        return F.relu(positive_distance - negative_distance + self.margin).mean()
