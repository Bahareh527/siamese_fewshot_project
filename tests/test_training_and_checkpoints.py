from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from siamese_fewshot import (
    CompactEncoder,
    ContrastiveLoss,
    PairDataset,
    PatternDataset,
    SiameseNetwork,
    load_checkpoint,
    save_checkpoint,
    train_contrastive_epoch,
)


def test_training_epoch_and_checkpoint_round_trip(tmp_path) -> None:
    dataset = PatternDataset(class_count=3, samples_per_class=4, image_size=16, seed=5)
    loader = DataLoader(PairDataset(dataset, length=12, seed=5), batch_size=4)
    model = SiameseNetwork(CompactEncoder(embedding_dim=8, base_channels=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_contrastive_epoch(model, loader, optimizer, ContrastiveLoss())
    assert loss >= 0

    checkpoint = save_checkpoint(
        tmp_path / "model.pt", model, optimizer, epoch=1, metadata={"seed": 5}
    )
    restored = SiameseNetwork(CompactEncoder(embedding_dim=8, base_channels=2))
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(checkpoint, restored, restored_optimizer)
    assert payload["epoch"] == 1
    assert payload["metadata"] == {"seed": 5}


def test_missing_checkpoint_is_not_silently_ignored(tmp_path) -> None:
    model = SiameseNetwork(CompactEncoder(embedding_dim=8, base_channels=2))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)
