from __future__ import annotations

import torch


def pairwise_squared_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Return a stable matrix of pairwise squared Euclidean distances."""
    squared_norms = embeddings.square().sum(dim=1, keepdim=True)
    distances = squared_norms + squared_norms.T - 2 * embeddings @ embeddings.T
    return distances.clamp_min(0)


def mine_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
    mode: str = "semi-hard",
) -> torch.Tensor:
    """Mine one informative triplet per valid anchor from a mini-batch."""
    if mode not in {"hard", "semi-hard"}:
        raise ValueError("mode must be 'hard' or 'semi-hard'.")
    distances = pairwise_squared_distances(embeddings.detach())
    labels = labels.view(-1)
    triplets: list[tuple[int, int, int]] = []
    for anchor in range(len(labels)):
        positives = torch.where(labels == labels[anchor])[0]
        positives = positives[positives != anchor]
        negatives = torch.where(labels != labels[anchor])[0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        positive = positives[distances[anchor, positives].argmax()]
        if mode == "hard":
            negative = negatives[distances[anchor, negatives].argmin()]
        else:
            positive_distance = distances[anchor, positive]
            negative_distances = distances[anchor, negatives]
            eligible = negatives[
                (negative_distances > positive_distance)
                & (negative_distances < positive_distance + margin)
            ]
            if len(eligible) == 0:
                continue
            negative = eligible[distances[anchor, eligible].argmin()]
        triplets.append((anchor, int(positive), int(negative)))
    if not triplets:
        return torch.empty((0, 3), dtype=torch.long, device=embeddings.device)
    return torch.tensor(triplets, dtype=torch.long, device=embeddings.device)
