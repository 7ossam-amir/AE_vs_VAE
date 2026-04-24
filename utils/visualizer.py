# utils/visualizer.py
"""
Visualization utilities for AE and VAE experiments.

Functions
─────────
plot_reconstructions()        – side-by-side original vs reconstructed
plot_loss_curves()            – training/validation loss over epochs
plot_latent_space_2d()        – 2-D PCA / UMAP scatter of latent codes
plot_latent_space_grid()      – VAE latent grid traversal (2 dims)
plot_generated_samples()      – images decoded from random z ~ N(0, I)
plot_denoising_comparison()   – noisy input → AE output vs clean target
plot_all_regions_comparison() – reconstruction quality bar-chart
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / Colab compatible
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

import tensorflow as tf

from configs.config import PLOT_DIR, ANATOMICAL_REGIONS, LATENT_DIM


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CMAP = "gray"

def _save(fig, filename: str):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


def _to_numpy(x):
    """Ensure x is a NumPy array in [0, 1]."""
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Reconstruction grid
# ---------------------------------------------------------------------------

def plot_reconstructions(
    originals: tf.Tensor,
    reconstructions: tf.Tensor,
    region: str,
    model_type: str = "AE",
    n: int = 8,
    filename: Optional[str] = None,
):
    """
    Displays n originals on the top row, reconstructions on the bottom.
    """
    originals      = _to_numpy(originals[:n])
    reconstructions = _to_numpy(reconstructions[:n])

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.5))
    fig.suptitle(f"{model_type} Reconstructions – {region}", fontsize=13, fontweight="bold")

    for i in range(n):
        axes[0, i].imshow(originals[i, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original",  fontsize=9, rotation=90, labelpad=4)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9, rotation=90, labelpad=4)

    filename = filename or f"{model_type}_{region}_reconstructions.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history,
    region: str,
    model_type: str = "AE",
    filename: Optional[str] = None,
):
    """
    Plot training + validation losses from a Keras History object.
    Handles both AE (single loss) and VAE (total / rec / kl).
    """
    hist  = history.history
    keys  = list(hist.keys())
    train_keys = [k for k in keys if not k.startswith("val_")]
    val_keys   = [f"val_{k}" for k in train_keys if f"val_{k}" in hist]

    n_plots = len(train_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(f"{model_type} Training Curves – {region}", fontsize=13, fontweight="bold")

    for ax, k in zip(axes, train_keys):
        ax.plot(hist[k], label="Train", linewidth=1.8)
        vk = f"val_{k}"
        if vk in hist:
            ax.plot(hist[vk], label="Val", linewidth=1.8, linestyle="--")
        ax.set_title(k.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    filename = filename or f"{model_type}_{region}_loss_curves.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Latent space (2-D)
# ---------------------------------------------------------------------------

def plot_latent_space_2d(
    latent_codes: np.ndarray,
    region: str,
    model_type: str = "AE",
    method: str = "pca",       # "pca" | "umap"
    labels: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
):
    """
    Project latent_codes to 2-D with PCA or UMAP and scatter-plot.

    Args:
        latent_codes: [N, latent_dim] array
        labels:       Optional integer class labels for colouring.
        method:       "pca" (always available) or "umap" (needs umap-learn).
    """
    from sklearn.decomposition import PCA

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latent_codes)
            method_label = "UMAP"
        except ImportError:
            print("  [warn] umap-learn not installed; falling back to PCA.")
            method = "pca"

    if method == "pca":
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(latent_codes)
        var = pca.explained_variance_ratio_
        method_label = f"PCA (var: {var[0]:.1%}, {var[1]:.1%})"

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10" if labels is not None else None,
        s=8, alpha=0.6,
    )
    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Class")
    ax.set_title(f"{model_type} Latent Space – {region}\n({method_label})", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)

    filename = filename or f"{model_type}_{region}_latent_{method}.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# VAE: latent grid traversal
# ---------------------------------------------------------------------------

def plot_latent_grid(
    vae,
    region: str,
    dim1: int = 0,
    dim2: int = 1,
    n_steps: int = 12,
    value_range: float = 3.0,
    filename: Optional[str] = None,
):
    """
    Traverse two latent dimensions over [−value_range, value_range],
    keeping all other dims at 0, and decode each point.
    """
    grid_vals = np.linspace(-value_range, value_range, n_steps)
    fig, axes = plt.subplots(n_steps, n_steps, figsize=(n_steps * 1.1, n_steps * 1.1))
    fig.suptitle(f"VAE Latent Grid – {region}  (dim{dim1} × dim{dim2})", fontsize=11, fontweight="bold")

    for i, v1 in enumerate(grid_vals):
        for j, v2 in enumerate(grid_vals):
            z = np.zeros((1, vae.latent_dim), dtype=np.float32)
            z[0, dim1] = v1
            z[0, dim2] = v2
            img = vae.decode(tf.constant(z)).numpy()[0, ..., 0]
            axes[i, j].imshow(np.clip(img, 0, 1), cmap=_CMAP, vmin=0, vmax=1)
            axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = filename or f"VAE_{region}_latent_grid_d{dim1}d{dim2}.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# VAE: prior sampling
# ---------------------------------------------------------------------------

def plot_generated_samples(
    vae,
    region: str,
    n: int = 16,
    filename: Optional[str] = None,
):
    """Sample n images from p(z) = N(0, I) and display them."""
    samples = _to_numpy(vae.generate(n))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"VAE Generated Samples – {region}", fontsize=12, fontweight="bold")
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(samples[i, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    filename = filename or f"VAE_{region}_generated_samples.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Denoising comparison
# ---------------------------------------------------------------------------

def plot_denoising_comparison(
    noisy: tf.Tensor,
    denoised: tf.Tensor,
    clean: tf.Tensor,
    region: str,
    n: int = 8,
    filename: Optional[str] = None,
):
    """Three-row grid: noisy → denoised → clean."""
    noisy    = _to_numpy(noisy[:n])
    denoised = _to_numpy(denoised[:n])
    clean    = _to_numpy(clean[:n])

    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 5))
    fig.suptitle(f"Denoising AE – {region}", fontsize=13, fontweight="bold")
    labels = ["Noisy Input", "AE Output", "Clean Target"]

    for row, (imgs, lbl) in enumerate(zip([noisy, denoised, clean], labels)):
        for col in range(n):
            axes[row, col].imshow(imgs[col, ..., 0], cmap=_CMAP, vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(lbl, fontsize=9, rotation=90, labelpad=4)

    filename = filename or f"AE_{region}_denoising.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Comparison bar-chart (MSE across all regions)
# ---------------------------------------------------------------------------

def plot_mse_comparison(
    mse_ae: dict,
    mse_vae: dict,
    filename: str = "mse_comparison_all_regions.png",
):
    """
    Bar chart comparing reconstruction MSE for AE vs VAE per region.

    Args:
        mse_ae:  {region: float}
        mse_vae: {region: float}
    """
    regions  = ANATOMICAL_REGIONS
    ae_vals  = [mse_ae.get(r, 0) for r in regions]
    vae_vals = [mse_vae.get(r, 0) for r in regions]

    x    = np.arange(len(regions))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, ae_vals,  w, label="AE",  color="#4C9BE8", edgecolor="white")
    ax.bar(x + w/2, vae_vals, w, label="VAE", color="#E86B4C", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=20, ha="right")
    ax.set_ylabel("Val MSE")
    ax.set_title("Reconstruction MSE – AE vs VAE (all regions)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Optional type annotation (Python 3.8 compat)
# ---------------------------------------------------------------------------
from typing import Optional
