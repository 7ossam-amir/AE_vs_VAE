"""Compatibility layer for legacy imports.

Prefer importing training utilities from `src.train`.
"""

from src.train import (
    KLAnnealingCallback,
    TrainingOutput,
    train_all_regions,
    train_autoencoder,
    train_vae,
)

__all__ = [
    "KLAnnealingCallback",
    "TrainingOutput",
    "train_all_regions",
    "train_autoencoder",
    "train_vae",
]

