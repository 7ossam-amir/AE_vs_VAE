# training/losses.py
"""
Loss functions for the Autoencoder and Variational Autoencoder.

AE  → MSE reconstruction loss
VAE → MSE reconstruction loss + KL divergence (with optional annealing)
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# AE loss
# ---------------------------------------------------------------------------

def reconstruction_loss(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    """
    Mean Squared Error per sample, averaged over the batch.

    Args:
        x_true: Ground-truth images  [B, H, W, C]
        x_pred: Reconstructed images [B, H, W, C]

    Returns:
        Scalar MSE loss.
    """
    # Reduce over spatial + channel dims, keep batch dim, then mean
    per_sample = tf.reduce_mean(tf.square(x_true - x_pred), axis=[1, 2, 3])
    return tf.reduce_mean(per_sample)


# ---------------------------------------------------------------------------
# VAE losses
# ---------------------------------------------------------------------------

def kl_divergence(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    """
    Closed-form KL divergence KL[ q(z|x) || p(z) ] for Gaussian q and p=N(0,I).

        KL = -0.5 * sum( 1 + log_var - mean^2 - exp(log_var) )

    Args:
        z_mean:    Mean of approximate posterior    [B, latent_dim]
        z_log_var: Log-variance of approximate post. [B, latent_dim]

    Returns:
        Scalar KL loss (averaged over batch and latent dims).
    """
    kl_per_dim = -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    # Sum over latent dims, mean over batch
    return tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))


def vae_loss(
    x_true:    tf.Tensor,
    x_pred:    tf.Tensor,
    z_mean:    tf.Tensor,
    z_log_var: tf.Tensor,
    kl_weight: float = 1.0,
) -> tf.Tensor:
    """
    Combined VAE loss = reconstruction_loss + kl_weight * kl_divergence.

    Args:
        x_true:     Ground-truth images              [B, H, W, C]
        x_pred:     Decoded / reconstructed images   [B, H, W, C]
        z_mean:     Encoder mean output              [B, latent_dim]
        z_log_var:  Encoder log-variance output      [B, latent_dim]
        kl_weight:  Scalar β weighting KL term (β-VAE formulation).
                    Use KL annealing by passing an increasing schedule.

    Returns:
        Scalar total loss.
    """
    rec_loss = reconstruction_loss(x_true, x_pred)
    kl_loss  = kl_divergence(z_mean, z_log_var)
    return rec_loss + kl_weight * kl_loss, rec_loss, kl_loss


# ---------------------------------------------------------------------------
# KL annealing schedule
# ---------------------------------------------------------------------------

class KLAnnealer:
    """
    Linearly ramps KL weight from 0.0 to `max_weight` over `n_epochs` epochs.
    After `n_epochs`, weight is fixed at `max_weight`.

    Usage:
        annealer = KLAnnealer(max_weight=1.0, n_epochs=20)
        for epoch in range(total_epochs):
            kl_w = annealer(epoch)
            # pass kl_w to vae_loss(...)
    """

    def __init__(self, max_weight: float = 1.0, n_epochs: int = 20):
        self.max_weight = max_weight
        self.n_epochs   = n_epochs

    def __call__(self, epoch: int) -> float:
        if self.n_epochs == 0:
            return self.max_weight
        return float(min(self.max_weight, self.max_weight * epoch / self.n_epochs))
