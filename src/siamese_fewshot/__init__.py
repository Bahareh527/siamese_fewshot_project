"""Siamese-network utilities for reproducible few-shot experiments."""

from .data import ImageFolderDataset, PairDataset, TripletDataset
from .episodes import Episode, EpisodeMetrics, evaluate_episodes, sample_episode
from .losses import ContrastiveLoss, TripletLoss
from .mining import mine_triplets, pairwise_squared_distances
from .models import CompactEncoder, SiameseNetwork
from .synthetic import PatternDataset
from .training import train_contrastive_epoch, train_triplet_epoch
from .utils import load_checkpoint, save_checkpoint, seed_everything

__all__ = [
    "CompactEncoder",
    "ContrastiveLoss",
    "Episode",
    "EpisodeMetrics",
    "ImageFolderDataset",
    "PairDataset",
    "PatternDataset",
    "SiameseNetwork",
    "TripletDataset",
    "TripletLoss",
    "evaluate_episodes",
    "load_checkpoint",
    "mine_triplets",
    "pairwise_squared_distances",
    "sample_episode",
    "save_checkpoint",
    "seed_everything",
    "train_contrastive_epoch",
    "train_triplet_epoch",
]
