"""Evaluation metrics and summarization helpers for trained models."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import tensorflow as tf

from .config import ExperimentConfig, runtime_config
from .data_processing import build_dataset


def _resolve_config(config: ExperimentConfig | None = None) -> ExperimentConfig:
    """Return runtime configuration, creating defaults when needed."""
    return config if config is not None else runtime_config()


def compute_mse(x_true: tf.Tensor, x_pred: tf.Tensor) -> float:
    """Compute mean squared reconstruction error."""
    mse = tf.reduce_mean(tf.square(x_true - x_pred))
    return float(mse.numpy())


def compute_ssim(x_true: tf.Tensor, x_pred: tf.Tensor) -> float:
    """Compute mean structural similarity index over a batch."""
    ssim_values = tf.image.ssim(x_true, x_pred, max_val=1.0)
    return float(tf.reduce_mean(ssim_values).numpy())


def evaluate_model(
    model: tf.keras.Model,
    region: str,
    model_type: str = "ae",
    config: ExperimentConfig | None = None,
) -> dict[str, float | str]:
    """Evaluate one model on the validation split for a region."""
    cfg = _resolve_config(config)
    val_ds = build_dataset(region=region, split="val", denoising=False, config=cfg)

    targets: list[tf.Tensor] = []
    predictions: list[tf.Tensor] = []
    for x_in, x_target in val_ds:
        x_pred = model(x_in, training=False)
        targets.append(x_target)
        predictions.append(x_pred)

    if not targets:
        raise RuntimeError(f"No validation batches produced for region '{region}'.")

    x_true = tf.concat(targets, axis=0)
    x_pred = tf.concat(predictions, axis=0)
    mse = compute_mse(x_true, x_pred)
    ssim = compute_ssim(x_true, x_pred)

    return {
        "region": region,
        "model_type": model_type,
        "mse": mse,
        "ssim": ssim,
    }


def evaluate_all_regions(
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: ExperimentConfig | None = None,
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Evaluate all models from a nested training results dictionary."""
    metrics: dict[str, dict[str, dict[str, float | str]]] = {}
    for region, model_runs in results.items():
        metrics[region] = {}
        for model_type, run_data in model_runs.items():
            model = run_data["model"]
            metrics[region][model_type] = evaluate_model(
                model=model,
                region=region,
                model_type=model_type,
                config=config,
            )
    return metrics


def metrics_to_dataframe(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | str]]],
) -> pd.DataFrame:
    """Flatten nested metrics to a tidy pandas DataFrame."""
    rows: list[dict[str, float | str]] = []
    for _, region_metrics in metrics.items():
        for _, metric_values in region_metrics.items():
            rows.append(dict(metric_values))
    if not rows:
        return pd.DataFrame(columns=["region", "model_type", "mse", "ssim"])
    return pd.DataFrame(rows).sort_values(["region", "model_type"]).reset_index(drop=True)


__all__ = [
    "compute_mse",
    "compute_ssim",
    "evaluate_all_regions",
    "evaluate_model",
    "metrics_to_dataframe",
]
