# AE vs VAE - DSAI 490 Assignment 1

This repository compares convolutional Autoencoders (AE) and Variational Autoencoders (VAE) on Medical MNIST, using one model per anatomical region.


## Project Structure

```text
dsai490_ae_vae/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|   |-- checkpoints/
|   `-- metadata/
|-- notebooks/
|   `-- experiment_notebook.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_processing.py
|   |-- losses.py
|   |-- metrics.py
|   |-- model.py
|   |-- train.py
|   `-- visualization.py
|-- tests/
|   |-- test_data_processing.py
|   `-- test_model.py
|-- configs/   (legacy compatibility wrappers)
|-- training/  (legacy compatibility wrappers)
|-- utils/     (legacy compatibility wrappers)
|-- requirements.txt
|-- pyproject.toml
|-- .flake8
`-- README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Data Layout

Place extracted Medical MNIST files in:

```text
data/raw/MedicalMNIST/
|-- AbdomenCT/
|-- BreastMRI/
|-- ChestCT/
|-- ChestXRay/
|-- Hand/
`-- HeadCT/
```

If your dataset is elsewhere, update `DATA_ROOT` in `src/config.py`.

## Training

```python
from src.train import train_autoencoder, train_vae

ae_result = train_autoencoder(region="ChestXRay", denoising=True)
vae_result = train_vae(region="ChestXRay")
```

Saved artifacts:

- Best weights: `models/checkpoints/<MODEL>_<REGION>_v<version>_best.weights.h5`
- Metadata: `models/metadata/<model_type>_<REGION>_v<version>.json`

Metadata includes timestamp, data root, validation split, hyperparameters, and training history.

## Evaluation

```python
from src.metrics import evaluate_model

metrics = evaluate_model(ae_result["model"], region="ChestXRay", model_type="ae")
print(metrics)
```

## Notebook

The notebook now lives at `notebooks/experiment_notebook.ipynb` and imports from `src.*`.

## Code Quality

Format:

```bash
black src tests
```

Lint:

```bash
flake8 src tests
pylint src
```

Tests:

```bash
pytest
```



