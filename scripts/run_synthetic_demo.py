from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from siamese_fewshot import (
    CompactEncoder,
    ContrastiveLoss,
    PairDataset,
    PatternDataset,
    SiameseNetwork,
    evaluate_episodes,
    seed_everything,
    train_contrastive_epoch,
)

ROOT = Path(__file__).resolve().parents[1]


def run_demo(epochs: int = 5) -> tuple[list[float], dict[str, float]]:
    """Train and evaluate a tiny synthetic experiment for software verification."""
    generator = seed_everything(42, deterministic=True)
    dataset = PatternDataset(class_count=6, samples_per_class=18, seed=42)
    pairs = PairDataset(dataset, length=384, seed=42)
    loader = DataLoader(pairs, batch_size=48, shuffle=True, generator=generator)
    model = SiameseNetwork(CompactEncoder(embedding_dim=32, base_channels=8))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = ContrastiveLoss(margin=1.0)
    losses = [
        train_contrastive_epoch(model, loader, optimizer, criterion) for _ in range(epochs)
    ]
    one_shot = evaluate_episodes(
        model, dataset, n_way=5, k_shot=1, queries_per_class=2, episodes=30, seed=100
    )
    four_shot = evaluate_episodes(
        model, dataset, n_way=5, k_shot=4, queries_per_class=2, episodes=30, seed=200
    )
    metrics = {
        "1-shot accuracy": one_shot.mean_accuracy,
        "1-shot 95% CI": one_shot.confidence_interval_95,
        "4-shot accuracy": four_shot.mean_accuracy,
        "4-shot 95% CI": four_shot.confidence_interval_95,
    }
    return losses, metrics


def main() -> None:
    losses, metrics = run_demo()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(range(1, len(losses) + 1), losses, marker="o", color="#4C78A8")
    axes[0].set(title="Contrastive training", xlabel="epoch", ylabel="mean loss")
    labels = ["1-shot", "4-shot"]
    values = [metrics["1-shot accuracy"], metrics["4-shot accuracy"]]
    errors = [metrics["1-shot 95% CI"], metrics["4-shot 95% CI"]]
    axes[1].bar(labels, values, yerr=errors, capsize=5, color=["#F58518", "#54A24B"])
    axes[1].axhline(0.2, color="#777777", linestyle="--", label="5-way chance")
    axes[1].set(
        title="Episodic evaluation (synthetic only)",
        ylabel="mean accuracy",
        ylim=(0, 1.05),
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    output = ROOT / "docs" / "figures" / "synthetic_demo.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print({key: round(value, 4) for key, value in metrics.items()})


if __name__ == "__main__":
    main()
