"""Compatibility layer for legacy imports.

Prefer importing evaluation utilities from `src.metrics`.
"""

from src.metrics import (
    compute_mse,
    compute_ssim,
    evaluate_all_regions,
    evaluate_model,
    metrics_to_dataframe,
)

__all__ = [
    "compute_mse",
    "compute_ssim",
    "evaluate_all_regions",
    "evaluate_model",
    "metrics_to_dataframe",
]

