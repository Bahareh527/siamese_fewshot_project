from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_contrastive_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> float:
    model.to(device)
    model.train()
    total_loss = 0.0
    sample_count = 0
    for first, second, labels in loader:
        first, second, labels = first.to(device), second.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        first_embedding, second_embedding = model.forward_pair(first, second)
        loss = criterion(first_embedding, second_embedding, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        sample_count += len(labels)
    if sample_count == 0:
        raise ValueError("The training loader is empty.")
    return total_loss / sample_count


def train_triplet_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> float:
    model.to(device)
    model.train()
    total_loss = 0.0
    sample_count = 0
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.forward_triplet(anchor, positive, negative)
        loss = criterion(*embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(anchor)
        sample_count += len(anchor)
    if sample_count == 0:
        raise ValueError("The training loader is empty.")
    return total_loss / sample_count
