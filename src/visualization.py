"""Plotting utilities for model reconstructions, latent spaces, and metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ExperimentConfig, runtime_config

_CMAP = "gray"


def _resolve_config(config: ExperimentConfig | None = None) -> ExperimentConfig:
    """Return runtime configuration, creating defaults when needed."""
    return config if config is not None else runtime_config()


def _to_numpy(array: tf.Tensor | np.ndarray) -> np.ndarray:
    """Convert TensorFlow tensors to clipped NumPy arrays."""
    if isinstance(array, tf.Tensor):
        array = array.numpy()
    return np.clip(array, 0.0, 1.0)


def _save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Save and close figure, returning its full path."""
    cfg = _resolve_config(config)
    destination_dir = Path(output_dir) if output_dir is not None else cfg.plot_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_reconstructions(
    originals: tf.Tensor,
    reconstructions: tf.Tensor,
    region: str,
    model_type: str = "AE",
    n: int = 8,
    filename: str | None = None,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Plot originals and reconstructions in a two-row grid."""
    originals_np = _to_numpy(originals[:n])
    recon_np = _to_numpy(reconstructions[:n])

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.5))
    fig.suptitle(f"{model_type} Reconstructions - {region}", fontsize=13, fontweight="bold")

    for index in range(n):
        axes[0, index].imshow(originals_np[index, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].imshow(recon_np[index, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
        axes[1, index].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9, rotation=90, labelpad=4)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9, rotation=90, labelpad=4)

    target_name = filename or f"{model_type}_{region}_reconstructions.png"
    return _save_figure(fig, target_name, output_dir=output_dir, config=config)


def plot_loss_curves(
    history: Any,
    region: str,
    model_type: str = "AE",
    filename: str | None = None,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Plot train/validation curves from a Keras history object."""
    history_data = history.history
    train_keys = [key for key in history_data.keys() if not key.startswith("val_")]
    subplot_count = len(train_keys)

    fig, axes = plt.subplots(1, subplot_count, figsize=(5 * subplot_count, 4))
    if subplot_count == 1:
        axes = [axes]

    fig.suptitle(f"{model_type} Training Curves - {region}", fontsize=13, fontweight="bold")
    for axis, train_key in zip(axes, train_keys):
        axis.plot(history_data[train_key], label="Train", linewidth=1.8)
        val_key = f"val_{train_key}"
        if val_key in history_data:
            axis.plot(history_data[val_key], label="Val", linewidth=1.8, linestyle="--")
        axis.set_title(train_key.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    target_name = filename or f"{model_type}_{region}_loss_curves.png"
    return _save_figure(fig, target_name, output_dir=output_dir, config=config)


def plot_latent_space_2d(
    latent_codes: np.ndarray,
    region: str,
    model_type: str = "AE",
    method: str = "pca",
    labels: np.ndarray | None = None,
    filename: str | None = None,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Project latent vectors to 2D using PCA or UMAP and plot a scatter chart."""
    from sklearn.decomposition import PCA

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latent_codes)
            method_label = "UMAP"
        except ImportError:
            method = "pca"

    if method == "pca":
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(latent_codes)
        explained = pca.explained_variance_ratio_
        method_label = f"PCA (var: {explained[0]:.1%}, {explained[1]:.1%})"

    fig, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10" if labels is not None else None,
        s=8,
        alpha=0.6,
    )
    if labels is not None:
        plt.colorbar(scatter, ax=axis, label="Class")
    axis.set_title(f"{model_type} Latent Space - {region}\n({method_label})", fontsize=12)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(True, alpha=0.2)

    target_name = filename or f"{model_type}_{region}_latent_{method}.png"
    return _save_figure(fig, target_name, output_dir=output_dir, config=config)


def plot_latent_grid(
    vae: Any,
    region: str,
    dim1: int = 0,
    dim2: int = 1,
    n_steps: int = 12,
    value_range: float = 3.0,
    filename: str | None = None,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Generate a 2D latent traversal grid for a trained VAE."""
    values = np.linspace(-value_range, value_range, n_steps)
    fig, axes = plt.subplots(n_steps, n_steps, figsize=(n_steps * 1.1, n_steps * 1.1))
    fig.suptitle(
        f"VAE Latent Grid - {region} (dim{dim1} x dim{dim2})",
        fontsize=11,
        fontweight="bold",
    )

    for row, value_1 in enumerate(values):
        for col, value_2 in enumerate(values):
            z = np.zeros((1, vae.latent_dim), dtype=np.float32)
            z[0, dim1] = value_1
            z[0, dim2] = value_2
            image = vae.decode(tf.constant(z)).numpy()[0, ..., 0]
            axes[row, col].imshow(np.clip(image, 0.0, 1.0), cmap=_CMAP, vmin=0, vmax=1)
            axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    target_name = filename or f"VAE_{region}_latent_grid_d{dim1}d{dim2}.png"
    return _save_figure(fig, target_name, output_dir=output_dir, config=config)


def plot_generated_samples(
    vae: Any,
    region: str,
    n: int = 16,
    filename: str | None = None,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Sample random latent vectors and plot generated outputs."""
    samples = _to_numpy(vae.generate(n))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"VAE Generated Samples - {region}", fontsize=12, fontweight="bold")
    axes = axes.flatten()

    for index in range(n):
        axes[index].imshow(samples[index, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
        axes[index].axis("off")
    for index in range(n, len(axes)):
        axes[index].axis("off")

    target_name = filename or f"VAE_{region}_generated_samples.png"
    return _save_figure(fig, target_name, output_dir=output_dir, config=config)


def plot_denoising_comparison(
    noisy: tf.Tensor,
    denoised: tf.Tensor,
    clean: tf.Tensor,
    region: str,
    n: int = 8,
    filename: str | None = None,
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Plot noisy inputs, denoised outputs, and clean targets."""
    noisy_np = _to_numpy(noisy[:n])
    denoised_np = _to_numpy(denoised[:n])
    clean_np = _to_numpy(clean[:n])

    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 5))
    fig.suptitle(f"Denoising AE - {region}", fontsize=13, fontweight="bold")
    row_labels = ["Noisy Input", "AE Output", "Clean Target"]

    for row_index, (images, label) in enumerate(zip((noisy_np, denoised_np, clean_np), row_labels)):
        for col_index in range(n):
            axes[row_index, col_index].imshow(images[col_index, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
            axes[row_index, col_index].axis("off")
        axes[row_index, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    target_name = filename or f"AE_{region}_denoising.png"
    return _save_figure(fig, target_name, output_dir=output_dir, config=config)


def plot_mse_comparison(
    mse_ae: dict[str, float],
    mse_vae: dict[str, float],
    filename: str = "mse_comparison_all_regions.png",
    output_dir: Path | str | None = None,
    config: ExperimentConfig | None = None,
) -> str:
    """Plot per-region MSE bars comparing AE and VAE reconstructions."""
    cfg = _resolve_config(config)
    regions = list(cfg.anatomical_regions)
    ae_values = [mse_ae.get(region, 0.0) for region in regions]
    vae_values = [mse_vae.get(region, 0.0) for region in regions]

    x_positions = np.arange(len(regions))
    width = 0.35

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x_positions - width / 2, ae_values, width, label="AE", color="#4C9BE8", edgecolor="white")
    axis.bar(x_positions + width / 2, vae_values, width, label="VAE", color="#E86B4C", edgecolor="white")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(regions, rotation=20, ha="right")
    axis.set_ylabel("Validation MSE")
    axis.set_title("Reconstruction MSE - AE vs VAE (all regions)", fontsize=13, fontweight="bold")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()

    return _save_figure(fig, filename, output_dir=output_dir, config=config)


__all__ = [
    "plot_denoising_comparison",
    "plot_generated_samples",
    "plot_latent_grid",
    "plot_latent_space_2d",
    "plot_loss_curves",
    "plot_mse_comparison",
    "plot_reconstructions",
]
