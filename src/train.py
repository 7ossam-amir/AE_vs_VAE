"""Training utilities and orchestration for AE and VAE experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from tensorflow import keras

from .config import ExperimentConfig, runtime_config
from .data_processing import build_dataset
from .losses import KLAnnealer
from .model import Autoencoder, VariationalAutoencoder

ModelType = Literal["ae", "vae", "both"]


class TrainingOutput(TypedDict):
    """Typed structure for one training run result."""

    model: keras.Model
    history: keras.callbacks.History
    checkpoint_path: str
    metadata_path: str


class KLAnnealingCallback(keras.callbacks.Callback):
    """Keras callback that updates VAE KL weight at epoch start."""

    def __init__(self, annealer: KLAnnealer) -> None:
        super().__init__()
        self.annealer = annealer

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        del logs
        new_weight = self.annealer(epoch)
        if hasattr(self.model, "kl_weight"):
            self.model.kl_weight.assign(new_weight)


def _resolve_config(config: ExperimentConfig | None = None) -> ExperimentConfig:
    """Return runtime configuration, creating defaults when needed."""
    return config if config is not None else runtime_config()


def _build_callbacks(
    model_name: str,
    monitor: str,
    cfg: ExperimentConfig,
    model_version: int,
) -> tuple[list[keras.callbacks.Callback], Path]:
    """Create standard callback set for all training jobs."""
    checkpoint_path = cfg.checkpoint_dir / f"{model_name}_v{model_version}_best.weights.h5"
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
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
    return callbacks, checkpoint_path


def _history_to_serializable(history: keras.callbacks.History) -> dict[str, list[float]]:
    """Convert keras History values to JSON-serializable float lists."""
    serializable: dict[str, list[float]] = {}
    for key, values in history.history.items():
        serializable[key] = [float(value) for value in values]
    return serializable


def _save_metadata(
    model_type: str,
    region: str,
    model_version: int,
    cfg: ExperimentConfig,
    history: keras.callbacks.History,
    checkpoint_path: Path,
    hyperparameters: dict[str, Any],
) -> Path:
    """Persist metadata for reproducibility and model lineage."""
    metadata_path = cfg.metadata_dir / f"{model_type}_{region}_v{model_version}.json"
    payload = {
        "model_type": model_type,
        "region": region,
        "version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "data_root": str(cfg.data_root),
        "anatomical_regions": list(cfg.anatomical_regions),
        "val_split": cfg.val_split,
        "hyperparameters": hyperparameters,
        "history": _history_to_serializable(history),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def train_autoencoder(
    region: str,
    denoising: bool = False,
    config: ExperimentConfig | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    model_version: int | None = None,
    verbose: int = 1,
) -> TrainingOutput:
    """Build, compile, and train one autoencoder for a region."""
    cfg = _resolve_config(config)
    epochs = cfg.ae_epochs if epochs is None else epochs
    learning_rate = cfg.learning_rate if learning_rate is None else learning_rate
    model_version = cfg.model_version if model_version is None else model_version

    train_ds = build_dataset(region=region, split="train", denoising=denoising, config=cfg)
    val_ds = build_dataset(region=region, split="val", denoising=denoising, config=cfg)

    model = Autoencoder(
        region=region,
        latent_dim=cfg.latent_dim,
        filters=cfg.ae_filters,
        input_shape=(cfg.image_size, cfg.image_size, cfg.num_channels),
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    for batch in train_ds.take(1):
        model(batch, training=False)

    callbacks, checkpoint_path = _build_callbacks(
        model_name=f"AE_{region}",
        monitor="val_loss",
        cfg=cfg,
        model_version=model_version,
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    metadata_path = _save_metadata(
        model_type="ae",
        region=region,
        model_version=model_version,
        cfg=cfg,
        history=history,
        checkpoint_path=checkpoint_path,
        hyperparameters={
            "denoising": denoising,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "latent_dim": cfg.latent_dim,
            "filters": list(cfg.ae_filters),
            "batch_size": cfg.batch_size,
        },
    )
    return {
        "model": model,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
    }


def train_vae(
    region: str,
    config: ExperimentConfig | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    kl_weight: float | None = None,
    kl_anneal_epochs: int | None = None,
    model_version: int | None = None,
    verbose: int = 1,
) -> TrainingOutput:
    """Build, compile, and train one VAE for a region."""
    cfg = _resolve_config(config)
    epochs = cfg.vae_epochs if epochs is None else epochs
    learning_rate = cfg.learning_rate if learning_rate is None else learning_rate
    kl_weight = cfg.kl_weight if kl_weight is None else kl_weight
    kl_anneal_epochs = cfg.kl_anneal_epochs if kl_anneal_epochs is None else kl_anneal_epochs
    model_version = cfg.model_version if model_version is None else model_version

    train_ds = build_dataset(region=region, split="train", denoising=False, config=cfg)
    val_ds = build_dataset(region=region, split="val", denoising=False, config=cfg)

    model = VariationalAutoencoder(
        region=region,
        latent_dim=cfg.latent_dim,
        filters=cfg.vae_filters,
        kl_weight=0.0,
        input_shape=(cfg.image_size, cfg.image_size, cfg.num_channels),
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    for batch in train_ds.take(1):
        model(batch, training=False)

    callbacks, checkpoint_path = _build_callbacks(
        model_name=f"VAE_{region}",
        monitor="val_total_loss",
        cfg=cfg,
        model_version=model_version,
    )
    callbacks.append(KLAnnealingCallback(KLAnnealer(max_weight=kl_weight, n_epochs=kl_anneal_epochs)))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    metadata_path = _save_metadata(
        model_type="vae",
        region=region,
        model_version=model_version,
        cfg=cfg,
        history=history,
        checkpoint_path=checkpoint_path,
        hyperparameters={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "kl_anneal_epochs": kl_anneal_epochs,
            "latent_dim": cfg.latent_dim,
            "filters": list(cfg.vae_filters),
            "batch_size": cfg.batch_size,
        },
    )
    return {
        "model": model,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
    }


def train_all_regions(
    model_type: ModelType = "both",
    denoising: bool = True,
    config: ExperimentConfig | None = None,
    verbose: int = 1,
) -> dict[str, dict[str, TrainingOutput]]:
    """Train requested model family for each anatomical region."""
    if model_type not in ("ae", "vae", "both"):
        raise ValueError("model_type must be one of: 'ae', 'vae', 'both'.")

    cfg = _resolve_config(config)
    results: dict[str, dict[str, TrainingOutput]] = {}

    for region in cfg.anatomical_regions:
        results[region] = {}
        if model_type in ("ae", "both"):
            results[region]["ae"] = train_autoencoder(
                region=region,
                denoising=denoising,
                config=cfg,
                verbose=verbose,
            )
        if model_type in ("vae", "both"):
            results[region]["vae"] = train_vae(
                region=region,
                config=cfg,
                verbose=verbose,
            )

    return results


__all__ = [
    "KLAnnealingCallback",
    "ModelType",
    "TrainingOutput",
    "train_all_regions",
    "train_autoencoder",
    "train_vae",
]
