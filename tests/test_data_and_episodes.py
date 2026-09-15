from __future__ import annotations

import torch
from torch import nn

from siamese_fewshot import PairDataset, PatternDataset, evaluate_episodes, sample_episode


class IdentityEncoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(inputs.flatten(1), dim=1)


def test_pair_sampling_is_deterministic_and_labelled() -> None:
    dataset = PatternDataset(class_count=4, samples_per_class=6, seed=1)
    pairs = PairDataset(dataset, length=20, positive_probability=0.5, seed=8)
    first = pairs[3]
    second = pairs[3]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2].item() in {0.0, 1.0}


def test_episode_support_and_queries_are_disjoint() -> None:
    dataset = PatternDataset(class_count=5, samples_per_class=8, seed=2)
    episode = sample_episode(
        dataset, n_way=4, k_shot=2, queries_per_class=2, seed=10
    )
    assert set(episode.support_indices).isdisjoint(episode.query_indices)
    assert episode.support_images.shape[0] == 8
    assert episode.query_images.shape[0] == 8


def test_evaluation_reports_uncertainty() -> None:
    dataset = PatternDataset(class_count=5, samples_per_class=8, seed=3)
    metrics = evaluate_episodes(
        IdentityEncoder(),
        dataset,
        n_way=4,
        k_shot=2,
        queries_per_class=2,
        episodes=6,
        seed=12,
    )
    assert 0 <= metrics.mean_accuracy <= 1
    assert metrics.standard_deviation >= 0
    assert metrics.confidence_interval_95 >= 0
    assert metrics.episodes == 6
