# configs/config.py
"""
Global hyperparameters and dataset settings for DSAI 490 Assignment 1.
Adjust DATA_ROOT to point at your extracted Medical MNIST folder.
"""

import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Set this to wherever you extracted MedicalMNIST.zip
# e.g. "/content/medical_mnist" (Colab) or a local absolute path
DATA_ROOT = "/content/medical_mnist"   # <-- UPDATE THIS

# The 6 anatomical classes present in Medical MNIST
ANATOMICAL_REGIONS = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "ChestXRay",
    "Hand",
    "HeadCT",
]

IMAGE_SIZE = 64          # Images are 64×64 grayscale
NUM_CHANNELS = 1

# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
SHUFFLE_BUFFER = 2000
PREFETCH = True

# Train / validation split (fraction of data kept for validation)
VAL_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

LATENT_DIM = 16           # Shared across AE and VAE per region

# AE encoder hidden channel widths (decoder mirrors these in reverse)
AE_FILTERS = [32, 64, 128]

# VAE uses the same channel progression
VAE_FILTERS = [32, 64, 128]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

AE_EPOCHS  = 50
VAE_EPOCHS = 60

LEARNING_RATE = 1e-3

# KL annealing: linearly ramp KL weight from 0 → KL_WEIGHT over KL_ANNEAL_EPOCHS
KL_WEIGHT        = 1.0
KL_ANNEAL_EPOCHS = 20

# Denoising AE: standard deviation of Gaussian noise added to inputs
NOISE_STDDEV = 0.1

# ---------------------------------------------------------------------------
# Checkpointing & outputs
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "outputs/checkpoints"
PLOT_DIR       = "outputs/plots"
LOG_DIR        = "outputs/logs"

# Create directories if running locally
for _d in (CHECKPOINT_DIR, PLOT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)
