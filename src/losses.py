"""Loss functions and schedules for AE and VAE training."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


def reconstruction_loss(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    """Compute mean squared reconstruction loss."""
    per_sample = tf.reduce_mean(tf.square(x_true - x_pred), axis=[1, 2, 3])
    return tf.reduce_mean(per_sample)


def kl_divergence(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    """Compute KL divergence between q(z|x) and standard normal prior."""
    per_dim = -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return tf.reduce_mean(tf.reduce_sum(per_dim, axis=1))


def vae_loss(
    x_true: tf.Tensor,
    x_pred: tf.Tensor,
    z_mean: tf.Tensor,
    z_log_var: tf.Tensor,
    kl_weight: float | tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute total, reconstruction, and KL losses for a VAE."""
    rec = reconstruction_loss(x_true, x_pred)
    kl = kl_divergence(z_mean, z_log_var)
    total = rec + kl_weight * kl
    return total, rec, kl


@dataclass(frozen=True)
class KLAnnealer:
    """Linear scheduler for KL weight annealing."""

    max_weight: float = 1.0
    n_epochs: int = 20

    def __call__(self, epoch: int) -> float:
        """Return the KL weight for the current epoch."""
        if self.n_epochs <= 0:
            return float(self.max_weight)
        scaled = self.max_weight * float(epoch) / float(self.n_epochs)
        return float(min(self.max_weight, scaled))


__all__ = ["KLAnnealer", "kl_divergence", "reconstruction_loss", "vae_loss"]
