"""Compatibility layer for legacy imports.

Prefer importing model implementations from `src.model`.
"""

from src.model import Sampling, VariationalAutoencoder, build_vae_decoder, build_vae_encoder

__all__ = ["Sampling", "VariationalAutoencoder", "build_vae_decoder", "build_vae_encoder"]

