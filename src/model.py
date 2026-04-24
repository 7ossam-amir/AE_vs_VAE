"""Model definitions for convolutional AE and VAE architectures."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import AE_FILTERS, IMAGE_SIZE, LATENT_DIM, NUM_CHANNELS, VAE_FILTERS
from .losses import vae_loss

Shape3D = tuple[int, int, int]


def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Apply one downsampling convolution block."""
    x = layers.Conv2D(filters, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def _deconv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Apply one upsampling transpose-convolution block."""
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_autoencoder_encoder(
    latent_dim: int = LATENT_DIM,
    filters: Sequence[int] = tuple(AE_FILTERS),
    input_shape: Shape3D = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
    name: str = "ae_encoder",
) -> tuple[keras.Model, Shape3D]:
    """Create the encoder used by the deterministic autoencoder."""
    inputs = keras.Input(shape=input_shape, name="encoder_input")
    x = inputs
    for channel_count in filters:
        x = _conv_block(x, filters=channel_count)

    spatial_shape = x.shape[1:]
    if any(dim is None for dim in spatial_shape):
        raise ValueError("Encoder spatial shape must be statically known.")
    pre_flat_shape: Shape3D = (
        int(spatial_shape[0]),
        int(spatial_shape[1]),
        int(spatial_shape[2]),
    )

    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    return keras.Model(inputs, z, name=name), pre_flat_shape


def build_autoencoder_decoder(
    pre_flat_shape: Shape3D,
    latent_dim: int = LATENT_DIM,
    filters: Sequence[int] = tuple(AE_FILTERS),
    output_channels: int = NUM_CHANNELS,
    name: str = "ae_decoder",
) -> keras.Model:
    """Create the decoder used by the deterministic autoencoder."""
    height, width, channels = pre_flat_shape
    inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(height * width * channels)(inputs)
    x = layers.Reshape((height, width, channels))(x)

    if len(filters) > 1:
        for channel_count in reversed(filters[:-1]):
            x = _deconv_block(x, filters=channel_count)
    x = _deconv_block(x, filters=filters[0])

    outputs = layers.Conv2D(
        output_channels,
        kernel_size=1,
        activation="sigmoid",
        name="reconstruction",
    )(x)
    return keras.Model(inputs, outputs, name=name)


class Autoencoder(keras.Model):
    """Convolutional autoencoder with custom train/test steps."""

    def __init__(
        self,
        region: str,
        latent_dim: int = LATENT_DIM,
        filters: Sequence[int] = tuple(AE_FILTERS),
        input_shape: Shape3D = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        **kwargs: object,
    ) -> None:
        super().__init__(name=f"AE_{region}", **kwargs)
        self.region = region
        self.latent_dim = latent_dim
        self.filters = tuple(filters)

        self.encoder, self._pre_flat_shape = build_autoencoder_encoder(
            latent_dim=latent_dim,
            filters=self.filters,
            input_shape=input_shape,
            name=f"{region}_encoder",
        )
        self.decoder = build_autoencoder_decoder(
            pre_flat_shape=self._pre_flat_shape,
            latent_dim=latent_dim,
            filters=self.filters,
            output_channels=input_shape[-1],
            name=f"{region}_decoder",
        )
        self._loss_tracker = keras.metrics.Mean(name="loss")

    @staticmethod
    def _unpack_batch(
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Normalize Keras batch format to `(input_tensor, target_tensor)`."""
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            x_in = tf.convert_to_tensor(data[0])
            x_target = tf.convert_to_tensor(data[1])
            return x_in, x_target
        x_in = tf.convert_to_tensor(data)
        return x_in, x_in

    def call(
        self,
        inputs: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through encoder and decoder."""
        if isinstance(inputs, (tuple, list)):
            x_noisy = tf.convert_to_tensor(inputs[0])
        else:
            x_noisy = tf.convert_to_tensor(inputs)
        z = self.encoder(x_noisy, training=training)
        return self.decoder(z, training=training)

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        """Encode image tensors into latent vectors."""
        return self.encoder(inputs, training=False)

    def decode(self, latent_vectors: tf.Tensor) -> tf.Tensor:
        """Decode latent vectors into reconstructed images."""
        return self.decoder(latent_vectors, training=False)

    def train_step(
        self,
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Perform one optimization step using reconstruction MSE."""
        x_in, x_target = self._unpack_batch(data)
        with tf.GradientTape() as tape:
            x_pred = self(x_in, training=True)
            loss = tf.reduce_mean(tf.square(x_target - x_pred))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(
        self,
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Compute validation loss for one batch."""
        x_in, x_target = self._unpack_batch(data)
        x_pred = self(x_in, training=False)
        loss = tf.reduce_mean(tf.square(x_target - x_pred))
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        """List metrics that should reset every epoch."""
        return [self._loss_tracker]

    def print_summary(self) -> None:
        """Print encoder and decoder summaries."""
        print(f"\n{'=' * 60}")
        print(f"Autoencoder summary | region: {self.region} | latent_dim: {self.latent_dim}")
        print(f"{'=' * 60}")
        self.encoder.summary()
        self.decoder.summary()


class Sampling(layers.Layer):
    """Reparameterization layer for VAE latent sampling."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Sample from N(mean, exp(log_var))."""
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_encoder(
    latent_dim: int = LATENT_DIM,
    filters: Sequence[int] = tuple(VAE_FILTERS),
    input_shape: Shape3D = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
    name: str = "vae_encoder",
) -> tuple[keras.Model, Shape3D]:
    """Create the probabilistic VAE encoder."""
    inputs = keras.Input(shape=input_shape, name="enc_input")
    x = inputs
    for channel_count in filters:
        x = _conv_block(x, filters=channel_count)

    spatial_shape = x.shape[1:]
    if any(dim is None for dim in spatial_shape):
        raise ValueError("VAE encoder spatial shape must be statically known.")
    pre_flat_shape: Shape3D = (
        int(spatial_shape[0]),
        int(spatial_shape[1]),
        int(spatial_shape[2]),
    )

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")((z_mean, z_log_var))

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name=name)
    return encoder, pre_flat_shape


def build_vae_decoder(
    pre_flat_shape: Shape3D,
    latent_dim: int = LATENT_DIM,
    filters: Sequence[int] = tuple(VAE_FILTERS),
    output_channels: int = NUM_CHANNELS,
    name: str = "vae_decoder",
) -> keras.Model:
    """Create the probabilistic VAE decoder."""
    height, width, channels = pre_flat_shape
    inputs = keras.Input(shape=(latent_dim,), name="dec_input")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(height * width * channels, activation="relu")(x)
    x = layers.Reshape((height, width, channels))(x)

    if len(filters) > 1:
        for channel_count in reversed(filters[:-1]):
            x = _deconv_block(x, filters=channel_count)
    x = _deconv_block(x, filters=filters[0])

    outputs = layers.Conv2D(
        output_channels,
        kernel_size=1,
        activation="sigmoid",
        name="reconstruction",
    )(x)
    return keras.Model(inputs, outputs, name=name)


class VariationalAutoencoder(keras.Model):
    """Variational autoencoder with KL annealing support."""

    def __init__(
        self,
        region: str,
        latent_dim: int = LATENT_DIM,
        filters: Sequence[int] = tuple(VAE_FILTERS),
        kl_weight: float = 1.0,
        input_shape: Shape3D = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        **kwargs: object,
    ) -> None:
        super().__init__(name=f"VAE_{region}", **kwargs)
        self.region = region
        self.latent_dim = latent_dim
        self.filters = tuple(filters)
        self.kl_weight = tf.Variable(float(kl_weight), trainable=False, name="kl_weight")

        self.encoder, self._pre_flat_shape = build_vae_encoder(
            latent_dim=latent_dim,
            filters=self.filters,
            input_shape=input_shape,
            name=f"{region}_vae_encoder",
        )
        self.decoder = build_vae_decoder(
            pre_flat_shape=self._pre_flat_shape,
            latent_dim=latent_dim,
            filters=self.filters,
            output_channels=input_shape[-1],
            name=f"{region}_vae_decoder",
        )
        self._total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self._rec_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self._kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @staticmethod
    def _unpack_batch(
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Normalize Keras batch format to `(input_tensor, target_tensor)`."""
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            x_in = tf.convert_to_tensor(data[0])
            x_target = tf.convert_to_tensor(data[1])
            return x_in, x_target
        x_in = tf.convert_to_tensor(data)
        return x_in, x_in

    def call(
        self,
        inputs: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass to generate reconstructed images."""
        if isinstance(inputs, (tuple, list)):
            x = tf.convert_to_tensor(inputs[0])
        else:
            x = tf.convert_to_tensor(inputs)
        _, _, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encode image tensors to mean, log-variance, and sampled latent code."""
        z_mean, z_log_var, z = self.encoder(inputs, training=False)
        return z_mean, z_log_var, z

    def decode(self, latent_vectors: tf.Tensor) -> tf.Tensor:
        """Decode latent vectors into reconstructed images."""
        return self.decoder(latent_vectors, training=False)

    def generate(self, n: int = 16) -> tf.Tensor:
        """Generate `n` samples from the latent prior."""
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decode(z)

    def _compute_losses(
        self,
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        training: bool,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute total, reconstruction, and KL losses."""
        x_in, x_target = self._unpack_batch(data)
        z_mean, z_log_var, z = self.encoder(x_in, training=training)
        x_pred = self.decoder(z, training=training)
        total, rec, kl = vae_loss(
            x_true=x_target,
            x_pred=x_pred,
            z_mean=z_mean,
            z_log_var=z_log_var,
            kl_weight=self.kl_weight,
        )
        return total, rec, kl

    def train_step(
        self,
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Perform one optimization step for the VAE."""
        with tf.GradientTape() as tape:
            total, rec, kl = self._compute_losses(data=data, training=True)
        gradients = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._total_loss_tracker.update_state(total)
        self._rec_loss_tracker.update_state(rec)
        self._kl_loss_tracker.update_state(kl)
        return {
            "total_loss": self._total_loss_tracker.result(),
            "rec_loss": self._rec_loss_tracker.result(),
            "kl_loss": self._kl_loss_tracker.result(),
        }

    def test_step(
        self,
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Compute validation losses for one batch."""
        total, rec, kl = self._compute_losses(data=data, training=False)
        self._total_loss_tracker.update_state(total)
        self._rec_loss_tracker.update_state(rec)
        self._kl_loss_tracker.update_state(kl)
        return {
            "total_loss": self._total_loss_tracker.result(),
            "rec_loss": self._rec_loss_tracker.result(),
            "kl_loss": self._kl_loss_tracker.result(),
        }

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        """List metrics that should reset every epoch."""
        return [self._total_loss_tracker, self._rec_loss_tracker, self._kl_loss_tracker]

    def print_summary(self) -> None:
        """Print encoder and decoder summaries."""
        print(f"\n{'=' * 60}")
        print(
            "VAE summary | "
            f"region: {self.region} | latent_dim: {self.latent_dim} | "
            f"kl_weight: {self.kl_weight.numpy():.3f}"
        )
        print(f"{'=' * 60}")
        self.encoder.summary()
        self.decoder.summary()


__all__ = [
    "Autoencoder",
    "Sampling",
    "VariationalAutoencoder",
    "build_autoencoder_decoder",
    "build_autoencoder_encoder",
    "build_vae_decoder",
    "build_vae_encoder",
]
