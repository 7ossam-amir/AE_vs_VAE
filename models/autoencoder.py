"""Compatibility layer for legacy imports.

Prefer importing model implementations from `src.model`.
"""

from src.model import (
    Autoencoder,
    build_autoencoder_decoder as build_decoder,
    build_autoencoder_encoder as build_encoder,
)

__all__ = ["Autoencoder", "build_decoder", "build_encoder"]

