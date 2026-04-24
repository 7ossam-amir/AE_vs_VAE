# models/autoencoder.py
"""
Convolutional Autoencoder (AE) for Medical MNIST.

Architecture
────────────
Encoder:  Input → [Conv2D(f, 3, stride=2) → BN → ReLU] × len(filters) → Flatten → Dense(latent_dim)
Decoder:  Dense(latent_dim) → Reshape → [ConvTranspose2D(f, 3, stride=2) → BN → ReLU] × len(filters) → Conv2D(1, 1) → Sigmoid

One AE instance is created PER anatomical region (separate weights).
"""

from typing import List, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from configs.config import IMAGE_SIZE, NUM_CHANNELS, LATENT_DIM, AE_FILTERS


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(
    latent_dim: int = LATENT_DIM,
    filters: List[int] = AE_FILTERS,
    name: str = "ae_encoder",
) -> keras.Model:
    """
    Convolutional encoder: image → latent vector.

    Returns a Keras functional model whose output is a flat latent vector
    of shape [B, latent_dim].
    """
    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="encoder_input")
    x   = inp

    for f in filters:
        x = layers.Conv2D(f, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Store the spatial shape before flattening for the decoder
    pre_flat_shape = x.shape[1:]   # (H', W', C_last)

    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, name="latent")(x)

    return keras.Model(inp, z, name=name), pre_flat_shape


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def build_decoder(
    pre_flat_shape,
    latent_dim: int = LATENT_DIM,
    filters: List[int] = AE_FILTERS,
    name: str = "ae_decoder",
) -> keras.Model:
    """
    Convolutional decoder: latent vector → reconstructed image.
    """
    h, w, c = pre_flat_shape

    inp = keras.Input(shape=(latent_dim,), name="decoder_input")
    x   = layers.Dense(h * w * c)(inp)
    x   = layers.Reshape((h, w, c))(x)

    for f in reversed(filters[:-1]):
        x = layers.Conv2DTranspose(f, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Final up-sampling to original resolution
    x = layers.Conv2DTranspose(filters[0], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer: single-channel sigmoid output in [0, 1]
    out = layers.Conv2D(NUM_CHANNELS, kernel_size=1, activation="sigmoid", name="reconstruction")(x)

    return keras.Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# Full AE (encoder + decoder as one tf.keras.Model)
# ---------------------------------------------------------------------------

class Autoencoder(keras.Model):
    """
    Full convolutional Autoencoder.

    Usage:
        ae = Autoencoder(region="ChestXRay")
        ae.compile(optimizer=..., loss=...)
        ae.fit(train_ds, ...)
    """

    def __init__(
        self,
        region: str,
        latent_dim: int = LATENT_DIM,
        filters: List[int] = AE_FILTERS,
        **kwargs,
    ):
        super().__init__(name=f"AE_{region}", **kwargs)
        self.region     = region
        self.latent_dim = latent_dim
        self.filters    = filters

        self.encoder, self._pre_flat_shape = build_encoder(
            latent_dim=latent_dim,
            filters=filters,
            name=f"{region}_encoder",
        )
        self.decoder = build_decoder(
            pre_flat_shape=self._pre_flat_shape,
            latent_dim=latent_dim,
            filters=filters,
            name=f"{region}_decoder",
        )

        # Loss trackers
        self._loss_tracker = keras.metrics.Mean(name="loss")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs, training: bool = False):
        """inputs can be a single tensor or (noisy, clean) tuple."""
        if isinstance(inputs, (tuple, list)):
            x_noisy, _ = inputs
        else:
            x_noisy = inputs
        z   = self.encoder(x_noisy, training=training)
        x_r = self.decoder(z,       training=training)
        return x_r

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode a batch of images to latent vectors."""
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode a batch of latent vectors to images."""
        return self.decoder(z, training=False)

    # ------------------------------------------------------------------
    # Custom train / test steps (uses our MSE reconstruction loss)
    # ------------------------------------------------------------------

    def train_step(self, data):
        if isinstance(data, (tuple, list)):
            x_in, x_target = data
        else:
            x_in = x_target = data

        with tf.GradientTape() as tape:
            x_pred = self(x_in, training=True)
            loss   = tf.reduce_mean(tf.square(x_target - x_pred))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, (tuple, list)):
            x_in, x_target = data
        else:
            x_in = x_target = data
        x_pred = self(x_in, training=False)
        loss   = tf.reduce_mean(tf.square(x_target - x_pred))
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    @property
    def metrics(self):
        return [self._loss_tracker]

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  Autoencoder – region: {self.region}")
        print(f"  Latent dim : {self.latent_dim}")
        print(f"{'='*60}")
        self.encoder.summary()
        self.decoder.summary()
