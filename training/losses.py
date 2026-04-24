"""Compatibility layer for legacy imports.

Prefer importing losses from `src.losses`.
"""

from src.losses import KLAnnealer, kl_divergence, reconstruction_loss, vae_loss

__all__ = ["KLAnnealer", "kl_divergence", "reconstruction_loss", "vae_loss"]

