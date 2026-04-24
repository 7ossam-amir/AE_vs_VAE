# utils/data_loader.py
"""
tf.data pipeline for Medical MNIST.

Expected folder layout after extracting MedicalMNIST.zip:
    <DATA_ROOT>/
        AbdomenCT/
            000001.jpg
            ...
        BreastMRI/
            ...
        ChestCT/ ...
        ChestXRay/ ...
        Hand/ ...
        HeadCT/ ...

Each sub-folder contains all images for that anatomical region.
"""

import os
import math
from typing import Tuple, Optional

import tensorflow as tf

from configs.config import (
    DATA_ROOT,
    ANATOMICAL_REGIONS,
    IMAGE_SIZE,
    NUM_CHANNELS,
    BATCH_SIZE,
    SHUFFLE_BUFFER,
    PREFETCH,
    VAL_SPLIT,
    NOISE_STDDEV,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _parse_image(file_path: tf.Tensor) -> tf.Tensor:
    """Read a JPEG/PNG file and return a normalised [0,1] float tensor."""
    raw = tf.io.read_file(file_path)
    # Decode as grayscale regardless of original channels
    image = tf.image.decode_jpeg(raw, channels=NUM_CHANNELS)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _add_noise(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return (noisy_image, clean_image) for denoising AE training."""
    noise = tf.random.normal(shape=tf.shape(image), stddev=NOISE_STDDEV)
    noisy = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy, image


def _identity(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return (image, image) for standard reconstruction training."""
    return image, image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_file_paths(region: str) -> list:
    """Return all image file paths for a given anatomical region."""
    region_dir = os.path.join(DATA_ROOT, region)
    if not os.path.isdir(region_dir):
        raise FileNotFoundError(
            f"Region directory not found: {region_dir}\n"
            f"Please set DATA_ROOT in configs/config.py correctly."
        )
    extensions = (".jpg", ".jpeg", ".png")
    paths = [
        os.path.join(region_dir, f)
        for f in sorted(os.listdir(region_dir))
        if f.lower().endswith(extensions)
    ]
    if not paths:
        raise RuntimeError(f"No images found in {region_dir}")
    return paths


def build_dataset(
    region: str,
    split: str = "train",        # "train" | "val" | "all"
    denoising: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for a single anatomical region.

    Args:
        region:    One of ANATOMICAL_REGIONS.
        split:     Which split to return.
        denoising: If True, the dataset yields (noisy, clean) tuples;
                   otherwise (clean, clean) tuples.
        seed:      Random seed for reproducible train/val split.

    Returns:
        Batched, prefetched tf.data.Dataset.
    """
    assert region in ANATOMICAL_REGIONS, f"Unknown region: {region}"
    assert split in ("train", "val", "all"), f"Unknown split: {split}"

    paths = get_file_paths(region)
    n_total = len(paths)
    n_val   = max(1, math.floor(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    # Deterministic split by sorting + taking slices
    if split == "train":
        paths = paths[:n_train]
    elif split == "val":
        paths = paths[n_train:]
    # "all" → use everything

    ds = tf.data.Dataset.from_tensor_slices(paths)

    if split == "train":
        ds = ds.shuffle(min(SHUFFLE_BUFFER, len(paths)), seed=seed)

    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    pair_fn = _add_noise if denoising else _identity
    ds = ds.map(pair_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if PREFETCH:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_all_region_datasets(
    split: str = "train",
    denoising: bool = False,
) -> dict:
    """
    Convenience function: returns {region: tf.data.Dataset} for all regions.
    """
    return {
        region: build_dataset(region, split=split, denoising=denoising)
        for region in ANATOMICAL_REGIONS
    }


def get_sample_batch(region: str, n: int = 16) -> tf.Tensor:
    """
    Return a single batch of n clean images (float32, [0,1]) for plotting.
    """
    ds = build_dataset(region, split="all", denoising=False)
    for clean, _ in ds.take(1):
        return clean[:n]


def dataset_info(region: str) -> dict:
    """Return a summary dict with image counts for a region."""
    paths   = get_file_paths(region)
    n_total = len(paths)
    n_val   = max(1, math.floor(n_total * VAL_SPLIT))
    return {
        "region":  region,
        "total":   n_total,
        "train":   n_total - n_val,
        "val":     n_val,
        "batches_train": math.ceil((n_total - n_val) / BATCH_SIZE),
    }
