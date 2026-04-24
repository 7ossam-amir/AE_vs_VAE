# DSAI 490 – Assignment 1: Representation Learning with Autoencoders

## Project Structure

```
dsai490_ae_vae/
├── configs/
│   └── config.py           # Hyperparameters and dataset paths
├── models/
│   ├── autoencoder.py      # AE architecture (per region)
│   └── vae.py              # VAE architecture (per region)
├── training/
│   ├── trainer.py          # Training loops for AE & VAE
│   └── losses.py           # Reconstruction + KL divergence losses
├── utils/
│   ├── data_loader.py      # tf.data pipeline for Medical MNIST
│   ├── visualizer.py       # Latent space & reconstruction plots
│   └── metrics.py          # SSIM, MSE evaluation helpers
├── experiment_notebook.ipynb   # Full experiment pipeline
└── README.md
```

## Dataset: Medical MNIST
- 6 anatomical regions: AbdomenCT, BreastMRI, ChestCT, ChestXRay, Hand, HeadCT
- Each 64×64 grayscale images
- Upload to Google Drive → mount in Colab → pass path in `config.py`

## Setup

```bash
pip install tensorflow matplotlib seaborn scikit-learn umap-learn
```

## Quick Start (Colab)

1. Upload `MedicalMNIST.zip` to Google Drive.
2. Mount Drive and unzip:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip /content/drive/MyDrive/MedicalMNIST.zip -d /content/medical_mnist
   ```
3. Set `DATA_ROOT = "/content/medical_mnist"` in `configs/config.py`.
4. Open and run `experiment_notebook.ipynb`.

## Key Design Decisions
- **Separate AE & VAE per anatomical region** – each region gets its own trained model.
- **tf.data pipeline** – efficient prefetching and augmentation.
- **Latent dim = 16** (configurable) – large enough for expressiveness, small enough to visualize with UMAP/PCA.
- **Denoising mode** – Gaussian noise is added to inputs during AE training.
