"""Unit tests for AE and VAE model APIs."""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from src.model import Autoencoder, VariationalAutoencoder


def test_autoencoder_forward_and_train_step() -> None:
    """Autoencoder should preserve image shape and expose training metrics."""
    model = Autoencoder(region="Hand", latent_dim=8, filters=(16, 32), input_shape=(64, 64, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    inputs = tf.random.uniform(shape=(4, 64, 64, 1))
    outputs = model(inputs, training=False)
    assert tuple(outputs.shape) == (4, 64, 64, 1)

    metrics = model.train_step((inputs, inputs))
    assert "loss" in metrics


def test_vae_forward_generate_and_train_step() -> None:
    """VAE should reconstruct, generate, and return expected loss keys."""
    model = VariationalAutoencoder(
        region="Hand",
        latent_dim=8,
        filters=(16, 32),
        input_shape=(64, 64, 1),
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    inputs = tf.random.uniform(shape=(4, 64, 64, 1))
    outputs = model(inputs, training=False)
    assert tuple(outputs.shape) == (4, 64, 64, 1)

    generated = model.generate(n=5)
    assert tuple(generated.shape) == (5, 64, 64, 1)

    metrics = model.train_step((inputs, inputs))
    assert {"total_loss", "rec_loss", "kl_loss"}.issubset(metrics.keys())

