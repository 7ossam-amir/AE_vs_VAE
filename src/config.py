"""Configuration objects and constants for the AE vs VAE project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Dataset
DATA_ROOT = "data/raw/MedicalMNIST"
ANATOMICAL_REGIONS = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "ChestXRay",
    "Hand",
    "HeadCT",
]
IMAGE_SIZE = 64
NUM_CHANNELS = 1

# tf.data settings
BATCH_SIZE = 64
SHUFFLE_BUFFER = 2000
PREFETCH = True
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# Model architecture
LATENT_DIM = 16
AE_FILTERS = [32, 64, 128]
VAE_FILTERS = [32, 64, 128]

# Training
AE_EPOCHS = 50
VAE_EPOCHS = 60
LEARNING_RATE = 1e-3
KL_WEIGHT = 1.0
KL_ANNEAL_EPOCHS = 20
NOISE_STDDEV = 0.1
MODEL_VERSION = 1

# Paths and outputs
MODELS_DIR = "models"
CHECKPOINT_DIR = "models/checkpoints"
METADATA_DIR = "models/metadata"
PLOT_DIR = "plots"
LOG_DIR = "outputs/logs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration used by data pipelines and training loops."""

    data_root: Path
    anatomical_regions: tuple[str, ...]
    image_size: int
    num_channels: int
    batch_size: int
    shuffle_buffer: int
    prefetch: bool
    val_split: float
    random_seed: int
    latent_dim: int
    ae_filters: tuple[int, ...]
    vae_filters: tuple[int, ...]
    ae_epochs: int
    vae_epochs: int
    learning_rate: float
    kl_weight: float
    kl_anneal_epochs: int
    noise_stddev: float
    model_version: int
    models_dir: Path
    checkpoint_dir: Path
    metadata_dir: Path
    plot_dir: Path
    log_dir: Path

    def ensure_directories(self) -> None:
        """Create all output directories required by the workflow."""
        for directory in (
            self.models_dir,
            self.checkpoint_dir,
            self.metadata_dir,
            self.plot_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def runtime_config() -> ExperimentConfig:
    """Build a config object from current module-level values."""
    config = ExperimentConfig(
        data_root=Path(DATA_ROOT),
        anatomical_regions=tuple(ANATOMICAL_REGIONS),
        image_size=IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        batch_size=BATCH_SIZE,
        shuffle_buffer=SHUFFLE_BUFFER,
        prefetch=PREFETCH,
        val_split=VAL_SPLIT,
        random_seed=RANDOM_SEED,
        latent_dim=LATENT_DIM,
        ae_filters=tuple(AE_FILTERS),
        vae_filters=tuple(VAE_FILTERS),
        ae_epochs=AE_EPOCHS,
        vae_epochs=VAE_EPOCHS,
        learning_rate=LEARNING_RATE,
        kl_weight=KL_WEIGHT,
        kl_anneal_epochs=KL_ANNEAL_EPOCHS,
        noise_stddev=NOISE_STDDEV,
        model_version=MODEL_VERSION,
        models_dir=Path(MODELS_DIR),
        checkpoint_dir=Path(CHECKPOINT_DIR),
        metadata_dir=Path(METADATA_DIR),
        plot_dir=Path(PLOT_DIR),
        log_dir=Path(LOG_DIR),
    )
    config.ensure_directories()
    return config


__all__ = [
    "AE_EPOCHS",
    "AE_FILTERS",
    "ANATOMICAL_REGIONS",
    "BATCH_SIZE",
    "CHECKPOINT_DIR",
    "DATA_ROOT",
    "ExperimentConfig",
    "IMAGE_SIZE",
    "KL_ANNEAL_EPOCHS",
    "KL_WEIGHT",
    "LATENT_DIM",
    "LEARNING_RATE",
    "LOG_DIR",
    "METADATA_DIR",
    "MODELS_DIR",
    "MODEL_VERSION",
    "NOISE_STDDEV",
    "NUM_CHANNELS",
    "PLOT_DIR",
    "PREFETCH",
    "RANDOM_SEED",
    "SHUFFLE_BUFFER",
    "VAL_SPLIT",
    "VAE_EPOCHS",
    "VAE_FILTERS",
    "runtime_config",
]
