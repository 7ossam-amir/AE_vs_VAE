"""Unit tests for data loading and split logic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
tf = pytest.importorskip("tensorflow")
from PIL import Image

from src.config import runtime_config
from src.data_processing import build_dataset, collect_region_file_paths, split_file_paths


def _write_dummy_images(region_dir: Path, n_images: int) -> None:
    """Create deterministic grayscale PNG test images."""
    rng = np.random.default_rng(seed=42)
    region_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n_images):
        image = (rng.random((64, 64)) * 255).astype(np.uint8)
        Image.fromarray(image, mode="L").save(region_dir / f"{index:04d}.png")


def _build_test_config(tmp_path: Path):
    """Create a runtime config object that points to temporary test paths."""
    base = runtime_config()
    config = replace(
        base,
        data_root=tmp_path,
        anatomical_regions=("Hand",),
        batch_size=4,
        val_split=0.2,
        models_dir=tmp_path / "models",
        checkpoint_dir=tmp_path / "models" / "checkpoints",
        metadata_dir=tmp_path / "models" / "metadata",
        plot_dir=tmp_path / "plots",
        log_dir=tmp_path / "logs",
    )
    config.ensure_directories()
    return config


def test_split_file_paths_uses_expected_train_val_counts(tmp_path: Path) -> None:
    """Split helper should allocate deterministic train and validation counts."""
    paths = [tmp_path / f"{index:04d}.png" for index in range(10)]
    train_paths = split_file_paths(paths, split="train", val_split=0.2)
    val_paths = split_file_paths(paths, split="val", val_split=0.2)
    assert len(train_paths) == 8
    assert len(val_paths) == 2


def test_build_dataset_returns_expected_sample_counts(tmp_path: Path) -> None:
    """Dataset builder should produce expected sample totals per split."""
    config = _build_test_config(tmp_path)
    _write_dummy_images(tmp_path / "Hand", n_images=10)

    collected = collect_region_file_paths(region="Hand", config=config)
    assert len(collected) == 10

    train_ds = build_dataset(region="Hand", split="train", denoising=False, config=config)
    val_ds = build_dataset(region="Hand", split="val", denoising=False, config=config)

    train_count = sum(int(batch_x.shape[0]) for batch_x, _ in train_ds)
    val_count = sum(int(batch_x.shape[0]) for batch_x, _ in val_ds)

    assert train_count == 8
    assert val_count == 2

    for batch_x, batch_y in train_ds.take(1):
        assert tuple(batch_x.shape[1:]) == (64, 64, 1)
        tf.debugging.assert_near(batch_x, batch_y)

