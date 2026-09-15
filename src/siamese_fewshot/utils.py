from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, and a reusable DataLoader generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save model state and the minimum information needed to resume an experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint strictly; missing or incompatible weights are never ignored."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Expected a checkpoint dictionary containing 'model_state_dict'.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        if "optimizer_state_dict" not in payload:
            raise ValueError("The checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
