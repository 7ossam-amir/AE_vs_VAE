"""Compatibility layer for legacy imports.

Prefer importing data pipeline utilities from `src.data_processing`.
"""

from src.data_processing import (
    build_all_region_datasets,
    build_dataset,
    collect_region_file_paths as get_file_paths,
    create_all_region_datasets,
    create_dataset,
    dataset_info,
    get_sample_batch,
    split_file_paths,
)

__all__ = [
    "build_all_region_datasets",
    "build_dataset",
    "create_all_region_datasets",
    "create_dataset",
    "dataset_info",
    "get_file_paths",
    "get_sample_batch",
    "split_file_paths",
]

