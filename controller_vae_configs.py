# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Configurations for ControllerVAE models."""

from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae import Config
from magenta.models.music_vae import MusicVAE
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

HParams = contrib_training.HParams
ds = tfp.distributions


CONFIG_MAP = {}


class ControllerVAE(MusicVAE):
  """Reduced-Dimensional Variational Autoencoder for controlling the original MusicVAE."""

  def encode(self, sequence, sequence_length, control_sequence=None):
    hparams = self.hparams
    z_size = hparams.z_size
    intermediate_size = hparams.intermediate_size
    original_z_size = hparams.original_z_size

    sequence = tf.to_float(sequence)
    if control_sequence is not None:
      control_sequence = tf.to_float(control_sequence)
      sequence = tf.concat([sequence, control_sequence], axis=-1)
    encoder_output = self.encoder.encode(sequence, sequence_length)

    mu = tf.layers.dense(
        encoder_output,
        original_z_size,
        name='encoder/mu',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    sigma = tf.layers.dense(
        encoder_output,
        original_z_size,
        activation=tf.nn.softplus,
        name='encoder/sigma',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    encoder_output = ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma).sample()

    for layer, size in enumerate(intermediate_size, 1):
      encoder_output = tf.layers.dense(
        encoder_output,
        size,
        activation=tf.nn.leaky_relu,
        name='controller/encoder/intermediate_%i' % layer,
        kernel_initializer=tf.keras.initializers.glorot_normal)

    mu = tf.layers.dense(
        encoder_output,
        z_size,
        name='controller/encoder/mu',
        kernel_initializer=tf.keras.initializers.glorot_normal)
    sigma = tf.layers.dense(
        encoder_output,
        z_size,
        activation=tf.nn.softplus,
        name='controller/encoder/sigma',
        kernel_initializer=tf.keras.initializers.glorot_normal)

    return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)


  def decode(self, z):
    hparams = self.hparams
    original_z_size = hparams.original_z_size
    intermediate_size = reversed(hparams.intermediate_size)
    decoder_output = z

    for layer, size in enumerate(intermediate_size, 1):
      decoder_output = tf.layers.dense(
        decoder_output,
        size,
        activation=tf.nn.leaky_relu,
        name='controller/decoder/intermediate_%i' % layer,
        kernel_initializer=tf.keras.initializers.glorot_normal)

    decoder_output = tf.layers.dense(
      decoder_output,
      original_z_size,
      activation=tf.nn.leaky_relu,
      name='controller/decoder/output',
      kernel_initializer=tf.keras.initializers.glorot_normal)

    return decoder_output


  def _compute_model_loss(
      self, input_sequence, output_sequence, sequence_length, control_sequence):
    hparams = self.hparams
    batch_size = hparams.batch_size

    input_sequence = tf.to_float(input_sequence)
    output_sequence = tf.to_float(output_sequence)

    max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

    input_sequence = input_sequence[:, :max_seq_len]

    if control_sequence is not None:
      control_depth = control_sequence.shape[-1]
      control_sequence = tf.to_float(control_sequence)
      control_sequence = control_sequence[:, :max_seq_len]
      # Shouldn't be necessary, but the slice loses shape information when
      # control depth is zero.
      control_sequence.set_shape([batch_size, None, control_depth])

    # The target/expected outputs.
    x_target = output_sequence[:, :max_seq_len]
    # Inputs to be fed to decoder, including zero padding for the initial input.
    x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                     [(0, 0), (1, 0), (0, 0)])
    x_length = tf.minimum(sequence_length, max_seq_len)

    # Either encode to get `z`, or do unconditional, decoder-only.
    if hparams.z_size:  # vae mode:
      q_z = self.encode(input_sequence, x_length, control_sequence)
      z = q_z.sample()
      z = self.decode(z)

      # Prior distribution.
      p_z = ds.MultivariateNormalDiag(
          loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

      # KL Divergence (nats)
      kl_div = ds.kl_divergence(q_z, p_z)

      # Concatenate the Z vectors to the inputs at each time step.
    else:  # unconditional, decoder-only generation
      kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
      z = None

    r_loss, metric_map = self.decoder.reconstruction_loss(
        x_input, x_target, x_length, z, control_sequence)[0:2]

    free_nats = hparams.free_bits * tf.math.log(2.0)
    kl_cost = tf.maximum(kl_div - free_nats, 0)

    beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step)))
            * hparams.max_beta)
    self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

    scalars_to_summarize = {
        'loss': self.loss,
        'losses/r_loss': r_loss,
        'losses/kl_loss': kl_cost,
        'losses/kl_bits': kl_div / tf.math.log(2.0),
        'losses/kl_beta': beta,
    }
    return metric_map, scalars_to_summarize


  def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
    if z is not None and int(z.shape[0]) != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0], n))

    if self.hparams.z_size and z is None:
      tf.logging.warning(
          'Sampling from conditional model without `z`. Using random `z`.')
      normal_shape = [n, self.hparams.z_size]
      normal_dist = tfp.distributions.Normal(
          loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape))
      z = normal_dist.sample()

    z = self.decode(z)

    return self.decoder.sample(n, max_length, z, c_input, **kwargs)



CONFIG_MAP['cat-drums_2bar_small_3dim'] = Config(
    model=ControllerVAE(lstm_models.BidirectionalLstmEncoder(),
                                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=3,
            intermediate_size=[128],
            original_z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)