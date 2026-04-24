"""Core package for AE vs VAE training and evaluation."""

from .config import ExperimentConfig, runtime_config
from .data_processing import (
    build_all_region_datasets,
    build_dataset,
    collect_region_file_paths,
    dataset_info,
    get_sample_batch,
    split_file_paths,
)
from .metrics import (
    compute_mse,
    compute_ssim,
    evaluate_all_regions,
    evaluate_model,
    metrics_to_dataframe,
)
from .model import Autoencoder, VariationalAutoencoder
from .train import train_all_regions, train_autoencoder, train_vae

__all__ = [
    "Autoencoder",
    "ExperimentConfig",
    "VariationalAutoencoder",
    "build_all_region_datasets",
    "build_dataset",
    "collect_region_file_paths",
    "compute_mse",
    "compute_ssim",
    "dataset_info",
    "evaluate_all_regions",
    "evaluate_model",
    "get_sample_batch",
    "metrics_to_dataframe",
    "runtime_config",
    "split_file_paths",
    "train_all_regions",
    "train_autoencoder",
    "train_vae",
]
