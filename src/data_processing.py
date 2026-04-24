"""Data loading and tf.data pipeline utilities for Medical MNIST."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence

import tensorflow as tf

from .config import ExperimentConfig, runtime_config

Split = Literal["train", "val", "all"]
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _resolve_config(config: ExperimentConfig | None = None) -> ExperimentConfig:
    """Return runtime configuration, creating defaults when needed."""
    return config if config is not None else runtime_config()


def collect_region_file_paths(
    region: str,
    config: ExperimentConfig | None = None,
    extensions: tuple[str, ...] = VALID_IMAGE_EXTENSIONS,
) -> list[Path]:
    """Collect all image file paths for a specific anatomical region."""
    cfg = _resolve_config(config)

    if region not in cfg.anatomical_regions:
        raise ValueError(f"Unknown region '{region}'. Expected one of: {cfg.anatomical_regions}")

    region_dir = cfg.data_root / region
    if not region_dir.is_dir():
        raise FileNotFoundError(
            f"Region directory not found: {region_dir}. "
            "Set DATA_ROOT in src/config.py to the extracted dataset path."
        )

    file_paths = sorted(
        path
        for path in region_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not file_paths:
        raise RuntimeError(f"No images found under: {region_dir}")
    return file_paths


def split_file_paths(
    file_paths: Sequence[Path],
    split: Split,
    val_split: float,
) -> list[Path]:
    """Split region file paths into train, validation, or all sets."""
    if split not in ("train", "val", "all"):
        raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'all'.")
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1). Received: {val_split}")
    if not file_paths:
        raise ValueError("file_paths must not be empty.")

    sorted_paths = sorted(file_paths)
    if split == "all":
        return list(sorted_paths)

    n_total = len(sorted_paths)
    if n_total == 1:
        return list(sorted_paths) if split == "train" else []

    n_val = max(1, math.floor(n_total * val_split))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    if split == "train":
        return list(sorted_paths[:n_train])
    return list(sorted_paths[n_train:])


def _decode_and_resize_image(
    file_path: tf.Tensor,
    image_size: int,
    num_channels: int,
) -> tf.Tensor:
    """Decode an image file and normalize it to [0, 1]."""
    image_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(image_bytes, channels=num_channels, expand_animations=False)
    image.set_shape((None, None, num_channels))
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _add_noise(image: tf.Tensor, noise_stddev: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Create a noisy-clean pair for denoising autoencoder training."""
    noise = tf.random.normal(shape=tf.shape(image), stddev=noise_stddev)
    noisy = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy, image


def _identity_pair(image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Create a clean-clean pair for standard reconstruction training."""
    return image, image


def build_dataset(
    region: str,
    split: Split = "train",
    denoising: bool = False,
    seed: int | None = None,
    config: ExperimentConfig | None = None,
) -> tf.data.Dataset:
    """Build a batched `tf.data.Dataset` for a region and split."""
    cfg = _resolve_config(config)
    seed = cfg.random_seed if seed is None else seed

    region_paths = collect_region_file_paths(region=region, config=cfg)
    selected_paths = split_file_paths(region_paths, split=split, val_split=cfg.val_split)

    if not selected_paths:
        raise RuntimeError(
            f"Split '{split}' for region '{region}' is empty. "
            "Adjust DATA_ROOT or VAL_SPLIT in src/config.py."
        )

    ds = tf.data.Dataset.from_tensor_slices([str(path) for path in selected_paths])

    if split == "train" and len(selected_paths) > 1:
        ds = ds.shuffle(min(cfg.shuffle_buffer, len(selected_paths)), seed=seed)

    ds = ds.map(
        lambda file_path: _decode_and_resize_image(
            file_path=file_path,
            image_size=cfg.image_size,
            num_channels=cfg.num_channels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if denoising:
        ds = ds.map(
            lambda image: _add_noise(image, noise_stddev=cfg.noise_stddev),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(_identity_pair, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(cfg.batch_size, drop_remainder=False)
    if cfg.prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_all_region_datasets(
    split: Split = "train",
    denoising: bool = False,
    config: ExperimentConfig | None = None,
) -> dict[str, tf.data.Dataset]:
    """Build datasets for all anatomical regions."""
    cfg = _resolve_config(config)
    return {
        region: build_dataset(region=region, split=split, denoising=denoising, config=cfg)
        for region in cfg.anatomical_regions
    }


def get_sample_batch(
    region: str,
    n: int = 16,
    config: ExperimentConfig | None = None,
) -> tf.Tensor:
    """Return up to `n` clean images from a region."""
    dataset = build_dataset(region=region, split="all", denoising=False, config=config)
    for clean_batch, _ in dataset.take(1):
        return clean_batch[:n]
    raise RuntimeError("Unable to retrieve a sample batch from the dataset.")


def dataset_info(region: str, config: ExperimentConfig | None = None) -> dict[str, int | str]:
    """Compute split and batch counts for one region."""
    cfg = _resolve_config(config)
    all_paths = collect_region_file_paths(region=region, config=cfg)
    train_paths = split_file_paths(all_paths, split="train", val_split=cfg.val_split)
    val_paths = split_file_paths(all_paths, split="val", val_split=cfg.val_split)

    return {
        "region": region,
        "total": len(all_paths),
        "train": len(train_paths),
        "val": len(val_paths),
        "batches_train": math.ceil(len(train_paths) / cfg.batch_size),
        "batches_val": math.ceil(len(val_paths) / cfg.batch_size),
    }


def create_dataset(
    region: str,
    split: Split = "train",
    denoising: bool = False,
    seed: int | None = None,
    config: ExperimentConfig | None = None,
) -> tf.data.Dataset:
    """Alias for `build_dataset` for naming preference compatibility."""
    return build_dataset(
        region=region,
        split=split,
        denoising=denoising,
        seed=seed,
        config=config,
    )


def create_all_region_datasets(
    split: Split = "train",
    denoising: bool = False,
    config: ExperimentConfig | None = None,
) -> dict[str, tf.data.Dataset]:
    """Alias for `build_all_region_datasets` for naming preference compatibility."""
    return build_all_region_datasets(split=split, denoising=denoising, config=config)


__all__ = [
    "Split",
    "VALID_IMAGE_EXTENSIONS",
    "build_all_region_datasets",
    "build_dataset",
    "collect_region_file_paths",
    "create_all_region_datasets",
    "create_dataset",
    "dataset_info",
    "get_sample_batch",
    "split_file_paths",
]
