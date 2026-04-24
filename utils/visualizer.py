"""Compatibility layer for legacy imports.

Prefer importing plotting utilities from `src.visualization`.
"""

from src.visualization import (
    plot_denoising_comparison,
    plot_generated_samples,
    plot_latent_grid,
    plot_latent_space_2d,
    plot_loss_curves,
    plot_mse_comparison,
    plot_reconstructions,
)

__all__ = [
    "plot_denoising_comparison",
    "plot_generated_samples",
    "plot_latent_grid",
    "plot_latent_space_2d",
    "plot_loss_curves",
    "plot_mse_comparison",
    "plot_reconstructions",
]

