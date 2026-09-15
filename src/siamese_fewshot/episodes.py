from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class LabelledDataset(Protocol):
    targets: list[int]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]: ...


@dataclass(frozen=True, slots=True)
class Episode:
    support_images: torch.Tensor
    support_labels: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    mean_accuracy: float
    standard_deviation: float
    confidence_interval_95: float
    episodes: int


def sample_episode(
    dataset: LabelledDataset,
    *,
    n_way: int,
    k_shot: int,
    queries_per_class: int,
    seed: int,
) -> Episode:
    """Sample disjoint support and query examples from selected classes."""
    if min(n_way, k_shot, queries_per_class) <= 0:
        raise ValueError("n_way, k_shot, and queries_per_class must be positive.")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        grouped[int(label)].append(index)
    eligible = [
        label for label, indices in grouped.items() if len(indices) >= k_shot + queries_per_class
    ]
    if len(eligible) < n_way:
        raise ValueError("Not enough eligible classes for the requested episode.")
    rng = random.Random(seed)
    selected = rng.sample(sorted(eligible), n_way)
    label_map = {label: episode_label for episode_label, label in enumerate(selected)}
    support_indices: list[int] = []
    query_indices: list[int] = []
    support_labels: list[int] = []
    query_labels: list[int] = []
    for label in selected:
        chosen = rng.sample(grouped[label], k_shot + queries_per_class)
        support_indices.extend(chosen[:k_shot])
        query_indices.extend(chosen[k_shot:])
        support_labels.extend([label_map[label]] * k_shot)
        query_labels.extend([label_map[label]] * queries_per_class)

    def images(indices: list[int]) -> torch.Tensor:
        return torch.stack([dataset[index][0] for index in indices])

    return Episode(
        support_images=images(support_indices),
        support_labels=torch.tensor(support_labels),
        query_images=images(query_indices),
        query_labels=torch.tensor(query_labels),
        support_indices=tuple(support_indices),
        query_indices=tuple(query_indices),
    )


def _encode(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    encode = getattr(model, "encode", model)
    return encode(inputs)


def evaluate_episodes(
    model: nn.Module,
    dataset: LabelledDataset,
    *,
    n_way: int = 5,
    k_shot: int = 1,
    queries_per_class: int = 2,
    episodes: int = 50,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> EpisodeMetrics:
    """Evaluate nearest-prototype accuracy over independently sampled episodes."""
    if episodes <= 1:
        raise ValueError("Use at least two episodes to estimate variability.")
    model = model.to(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for episode_index in range(episodes):
            episode = sample_episode(
                dataset,
                n_way=n_way,
                k_shot=k_shot,
                queries_per_class=queries_per_class,
                seed=seed + episode_index,
            )
            support = _encode(model, episode.support_images.to(device))
            queries = _encode(model, episode.query_images.to(device))
            support_labels = episode.support_labels.to(device)
            prototypes = torch.stack(
                [support[support_labels == label].mean(dim=0) for label in range(n_way)]
            )
            predictions = torch.cdist(queries, prototypes).argmin(dim=1).cpu()
            scores.append((predictions == episode.query_labels).float().mean().item())
    values = torch.tensor(scores)
    standard_deviation = values.std(unbiased=True).item()
    return EpisodeMetrics(
        mean_accuracy=values.mean().item(),
        standard_deviation=standard_deviation,
        confidence_interval_95=1.96 * standard_deviation / math.sqrt(episodes),
        episodes=episodes,
    )
