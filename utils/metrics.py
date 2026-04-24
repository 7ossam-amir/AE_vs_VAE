# utils/metrics.py
"""
Evaluation metrics for reconstruction quality.

Functions
─────────
compute_mse()      – Mean Squared Error
compute_ssim()     – Structural Similarity Index (via TensorFlow)
evaluate_model()   – Run a full val-set evaluation for AE or VAE
"""

import numpy as np
import tensorflow as tf

from utils.data_loader import build_dataset


def compute_mse(x_true: tf.Tensor, x_pred: tf.Tensor) -> float:
    """Pixel-wise MSE averaged over all images and spatial dims."""
    mse = tf.reduce_mean(tf.square(x_true - x_pred))
    return float(mse.numpy())


def compute_ssim(x_true: tf.Tensor, x_pred: tf.Tensor) -> float:
    """
    Mean SSIM over the batch.
    Requires images in [0, 1] with shape [B, H, W, 1].
    """
    ssim_vals = tf.image.ssim(x_true, x_pred, max_val=1.0)
    return float(tf.reduce_mean(ssim_vals).numpy())


def evaluate_model(model, region: str, model_type: str = "ae") -> dict:
    """
    Evaluate AE or VAE on the validation split and return metrics.

    Args:
        model:      Trained Autoencoder or VariationalAutoencoder.
        region:     Anatomical region name.
        model_type: "ae" or "vae" (affects how the model is called).

    Returns:
        dict with keys: mse, ssim, region, model_type
    """
    val_ds = build_dataset(region, split="val", denoising=False)

    all_true, all_pred = [], []

    for x_in, x_target in val_ds:
        x_pred = model(x_in, training=False)
        all_true.append(x_target)
        all_pred.append(x_pred)

    x_true = tf.concat(all_true, axis=0)
    x_pred = tf.concat(all_pred, axis=0)

    mse  = compute_mse(x_true, x_pred)
    ssim = compute_ssim(x_true, x_pred)

    print(f"  [{model_type.upper()} | {region}]  MSE: {mse:.5f}  |  SSIM: {ssim:.4f}")
    return {"region": region, "model_type": model_type, "mse": mse, "ssim": ssim}


def evaluate_all_regions(results: dict) -> dict:
    """
    Evaluate all trained models stored in the results dict from trainer.py.

    Args:
        results: results[region]["ae" | "vae"] = {"model": ..., "history": ...}

    Returns:
        metrics[region]["ae" | "vae"] = {"mse": ..., "ssim": ...}
    """
    metrics = {}
    for region, models in results.items():
        metrics[region] = {}
        for model_type, run in models.items():
            m = evaluate_model(run["model"], region, model_type=model_type)
            metrics[region][model_type] = m
    return metrics
