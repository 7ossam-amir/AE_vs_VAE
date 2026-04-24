# training/trainer.py
"""
Training utilities for AE and VAE.

Provides:
  - train_autoencoder()   – trains one AE for a given region
  - train_vae()           – trains one VAE for a given region, with KL annealing
  - train_all_regions()   – loops over all 6 regions and trains both model types
  - History utilities for later plotting
"""

import os
from typing import Optional, Dict

import tensorflow as tf
from tensorflow import keras

from configs.config import (
    ANATOMICAL_REGIONS,
    AE_EPOCHS, VAE_EPOCHS,
    LEARNING_RATE,
    KL_WEIGHT, KL_ANNEAL_EPOCHS,
    CHECKPOINT_DIR, LOG_DIR,
)
from models.autoencoder import Autoencoder
from models.vae import VariationalAutoencoder
from training.losses import KLAnnealer
from utils.data_loader import build_dataset


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _get_callbacks(model_name: str, monitor: str = "val_loss") -> list:
    """Standard callback set: early stopping + model checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.weights.h5")
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# KL annealing callback
# ---------------------------------------------------------------------------

class KLAnnealingCallback(keras.callbacks.Callback):
    """Updates vae.kl_weight at the start of each epoch."""

    def __init__(self, annealer: KLAnnealer):
        super().__init__()
        self.annealer = annealer

    def on_epoch_begin(self, epoch, logs=None):
        new_weight = self.annealer(epoch)
        self.model.kl_weight.assign(new_weight)


# ---------------------------------------------------------------------------
# AE trainer
# ---------------------------------------------------------------------------

def train_autoencoder(
    region: str,
    denoising: bool = False,
    epochs: int = AE_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    verbose: int = 1,
) -> Dict:
    """
    Build, compile, and train an Autoencoder for one anatomical region.

    Args:
        region:        One of the 6 Medical MNIST region names.
        denoising:     If True, corrupt inputs with Gaussian noise during training.
        epochs:        Maximum training epochs.
        learning_rate: Initial Adam learning rate.
        verbose:       Keras verbosity (0/1/2).

    Returns:
        dict with keys 'model' (Autoencoder) and 'history' (keras History).
    """
    print(f"\n{'─'*55}")
    print(f"  Training AE  |  region: {region}  |  denoising: {denoising}")
    print(f"{'─'*55}")

    # Data
    train_ds = build_dataset(region, split="train", denoising=denoising)
    val_ds   = build_dataset(region, split="val",   denoising=denoising)

    # Model
    ae = Autoencoder(region=region)
    ae.compile(optimizer=keras.optimizers.Adam(learning_rate))

    # Build graph by passing a dummy batch
    for batch in train_ds.take(1):
        ae(batch, training=False)

    if verbose:
        ae.print_summary()

    # Train
    callbacks = _get_callbacks(f"AE_{region}")
    history   = ae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    return {"model": ae, "history": history}


# ---------------------------------------------------------------------------
# VAE trainer
# ---------------------------------------------------------------------------

def train_vae(
    region: str,
    epochs: int = VAE_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    kl_weight: float = KL_WEIGHT,
    kl_anneal_epochs: int = KL_ANNEAL_EPOCHS,
    verbose: int = 1,
) -> Dict:
    """
    Build, compile, and train a VAE for one anatomical region.

    Returns:
        dict with keys 'model' (VariationalAutoencoder) and 'history'.
    """
    print(f"\n{'─'*55}")
    print(f"  Training VAE  |  region: {region}  |  β={kl_weight}")
    print(f"{'─'*55}")

    # Data (no noise — VAE learns stochastic encoding natively)
    train_ds = build_dataset(region, split="train", denoising=False)
    val_ds   = build_dataset(region, split="val",   denoising=False)

    # Model
    vae = VariationalAutoencoder(region=region, kl_weight=0.0)  # start at 0 for annealing
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate))

    # Build graph
    for batch in train_ds.take(1):
        vae(batch, training=False)

    if verbose:
        vae.print_summary()

    # Callbacks
    annealer  = KLAnnealer(max_weight=kl_weight, n_epochs=kl_anneal_epochs)
    callbacks = _get_callbacks(f"VAE_{region}", monitor="val_total_loss") + [
        KLAnnealingCallback(annealer)
    ]

    history = vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    return {"model": vae, "history": history}


# ---------------------------------------------------------------------------
# Train all regions
# ---------------------------------------------------------------------------

def train_all_regions(
    model_type: str = "both",   # "ae" | "vae" | "both"
    denoising: bool = True,
    verbose: int = 1,
) -> Dict:
    """
    Train AE and/or VAE for every anatomical region.

    Returns:
        Nested dict:  results[region]["ae"] = {"model": ..., "history": ...}
                      results[region]["vae"] = {"model": ..., "history": ...}
    """
    assert model_type in ("ae", "vae", "both")
    results = {}

    for region in ANATOMICAL_REGIONS:
        results[region] = {}

        if model_type in ("ae", "both"):
            results[region]["ae"] = train_autoencoder(
                region, denoising=denoising, verbose=verbose
            )

        if model_type in ("vae", "both"):
            results[region]["vae"] = train_vae(
                region, verbose=verbose
            )

    print("\n✓ All regions trained successfully.")
    return results
