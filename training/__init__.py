"""Legacy training package compatibility exports."""

from .losses import KLAnnealer, kl_divergence, reconstruction_loss, vae_loss
from .trainer import (
    KLAnnealingCallback,
    TrainingOutput,
    train_all_regions,
    train_autoencoder,
    train_vae,
)

__all__ = [
    "KLAnnealer",
    "KLAnnealingCallback",
    "TrainingOutput",
    "kl_divergence",
    "reconstruction_loss",
    "train_all_regions",
    "train_autoencoder",
    "train_vae",
    "vae_loss",
]
