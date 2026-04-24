# models/vae.py
"""
Variational Autoencoder (VAE) for Medical MNIST.

Architecture
────────────
Encoder:  Input → ConvBlocks → Flatten → Dense → (z_mean, z_log_var)
Sampling: z = z_mean + eps * exp(0.5 * z_log_var)   [reparameterization trick]
Decoder:  z → Dense → Reshape → ConvTransposeBlocks → Sigmoid output

One VAE instance is created PER anatomical region (separate weights).
"""

from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from configs.config import IMAGE_SIZE, NUM_CHANNELS, LATENT_DIM, VAE_FILTERS


# ---------------------------------------------------------------------------
# Reparameterization sampling layer
# ---------------------------------------------------------------------------

class Sampling(layers.Layer):
    """
    Reparameterization trick:
        z = mean + eps * std     where eps ~ N(0, I)

    This allows gradients to flow through the stochastic node.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch  = tf.shape(z_mean)[0]
        dim    = tf.shape(z_mean)[1]
        eps    = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_vae_encoder(
    latent_dim: int = LATENT_DIM,
    filters: List[int] = VAE_FILTERS,
    name: str = "vae_encoder",
):
    """
    Probabilistic encoder: image → (z_mean, z_log_var, z_sample).
    """
    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="enc_input")
    x   = inp

    for f in filters:
        x = layers.Conv2D(f, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    pre_flat_shape = x.shape[1:]          # (H', W', C_last)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling(name="z")([z_mean, z_log_var])

    model = keras.Model(inp, [z_mean, z_log_var, z], name=name)
    return model, pre_flat_shape


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def build_vae_decoder(
    pre_flat_shape,
    latent_dim: int = LATENT_DIM,
    filters: List[int] = VAE_FILTERS,
    name: str = "vae_decoder",
) -> keras.Model:
    """
    Probabilistic decoder: latent sample → reconstructed image.
    """
    h, w, c = pre_flat_shape

    inp = keras.Input(shape=(latent_dim,), name="dec_input")
    x   = layers.Dense(256, activation="relu")(inp)
    x   = layers.Dense(h * w * c, activation="relu")(x)
    x   = layers.Reshape((h, w, c))(x)

    for f in reversed(filters[:-1]):
        x = layers.Conv2DTranspose(f, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters[0], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out = layers.Conv2D(NUM_CHANNELS, kernel_size=1, activation="sigmoid", name="reconstruction")(x)

    return keras.Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class VariationalAutoencoder(keras.Model):
    """
    Full VAE with β-weighting and KL annealing support.

    Usage:
        vae = VariationalAutoencoder(region="HeadCT")
        vae.compile(optimizer=keras.optimizers.Adam(1e-3))
        vae.fit(train_ds, epochs=60, callbacks=[...])
    """

    def __init__(
        self,
        region: str,
        latent_dim: int = LATENT_DIM,
        filters: List[int] = VAE_FILTERS,
        kl_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=f"VAE_{region}", **kwargs)
        self.region     = region
        self.latent_dim = latent_dim
        self.filters    = filters

        # kl_weight is a tf.Variable so we can update it externally (annealing)
        self.kl_weight = tf.Variable(float(kl_weight), trainable=False, name="kl_weight")

        self.encoder, self._pre_flat_shape = build_vae_encoder(
            latent_dim=latent_dim,
            filters=filters,
            name=f"{region}_vae_encoder",
        )
        self.decoder = build_vae_decoder(
            pre_flat_shape=self._pre_flat_shape,
            latent_dim=latent_dim,
            filters=filters,
            name=f"{region}_vae_decoder",
        )

        # Metric trackers
        self._total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self._rec_loss_tracker   = keras.metrics.Mean(name="rec_loss")
        self._kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs, training: bool = False):
        if isinstance(inputs, (tuple, list)):
            x, _ = inputs
        else:
            x = inputs
        z_mean, z_log_var, z = self.encoder(x, training=training)
        x_recon              = self.decoder(z,     training=training)
        return x_recon

    def encode(self, x: tf.Tensor):
        """Returns (z_mean, z_log_var, z_sample)."""
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        return self.decoder(z, training=False)

    def generate(self, n: int = 16) -> tf.Tensor:
        """Sample n images from the prior p(z) = N(0, I)."""
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decode(z)

    # ------------------------------------------------------------------
    # Custom train / test steps
    # ------------------------------------------------------------------

    def _compute_losses(self, data, training: bool):
        if isinstance(data, (tuple, list)):
            x_in, x_target = data
        else:
            x_in = x_target = data

        z_mean, z_log_var, z = self.encoder(x_in, training=training)
        x_pred               = self.decoder(z,     training=training)

        # Reconstruction: pixel-wise MSE averaged over spatial dims & batch
        rec_loss = tf.reduce_mean(tf.reduce_mean(tf.square(x_target - x_pred), axis=[1, 2, 3]))

        # KL divergence: KL[ q(z|x) || N(0,I) ]
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = rec_loss + self.kl_weight * kl_loss
        return total_loss, rec_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total, rec, kl = self._compute_losses(data, training=True)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._total_loss_tracker.update_state(total)
        self._rec_loss_tracker.update_state(rec)
        self._kl_loss_tracker.update_state(kl)

        return {
            "total_loss": self._total_loss_tracker.result(),
            "rec_loss":   self._rec_loss_tracker.result(),
            "kl_loss":    self._kl_loss_tracker.result(),
        }

    def test_step(self, data):
        total, rec, kl = self._compute_losses(data, training=False)
        self._total_loss_tracker.update_state(total)
        self._rec_loss_tracker.update_state(rec)
        self._kl_loss_tracker.update_state(kl)
        return {
            "total_loss": self._total_loss_tracker.result(),
            "rec_loss":   self._rec_loss_tracker.result(),
            "kl_loss":    self._kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [self._total_loss_tracker, self._rec_loss_tracker, self._kl_loss_tracker]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  VAE – region: {self.region}")
        print(f"  Latent dim : {self.latent_dim}  |  KL weight: {self.kl_weight.numpy():.3f}")
        print(f"{'='*60}")
        self.encoder.summary()
        self.decoder.summary()
