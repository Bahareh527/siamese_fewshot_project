from __future__ import annotations

import pytest
import torch

from siamese_fewshot import (
    CompactEncoder,
    ContrastiveLoss,
    SiameseNetwork,
    TripletLoss,
    mine_triplets,
    pairwise_squared_distances,
)
from siamese_fewshot.models import resnet18_encoder


def test_contrastive_loss_rewards_close_positive_pairs() -> None:
    criterion = ContrastiveLoss(margin=1.0)
    anchor = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    other = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([1.0, 0.0])
    assert criterion(anchor, other, labels).item() == pytest.approx(0.0)


def test_triplet_loss_has_expected_margin() -> None:
    criterion = TripletLoss(margin=0.5)
    anchor = torch.tensor([[0.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[2.0, 0.0]])
    assert criterion(anchor, positive, negative).item() == pytest.approx(0.0)


def test_encoder_returns_normalized_embeddings() -> None:
    model = SiameseNetwork(CompactEncoder(embedding_dim=12, base_channels=4))
    embeddings = model(torch.rand(3, 3, 32, 32))
    assert embeddings.shape == (3, 12)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(3), atol=1e-5)


def test_pairwise_distances_are_symmetric() -> None:
    distances = pairwise_squared_distances(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
    assert torch.allclose(distances, distances.T)
    assert distances[0, 1].item() == pytest.approx(25.0)


def test_hard_mining_returns_valid_labels() -> None:
    embeddings = torch.tensor([[0.0], [0.4], [1.0], [1.2]])
    labels = torch.tensor([0, 0, 1, 1])
    triplets = mine_triplets(embeddings, labels, mode="hard")
    assert triplets.shape == (4, 3)
    for anchor, positive, negative in triplets:
        assert labels[anchor] == labels[positive]
        assert labels[anchor] != labels[negative]


def test_resnet_requires_an_explicit_pretrained_source() -> None:
    with pytest.raises(ValueError, match="Pretrained weights"):
        resnet18_encoder()
