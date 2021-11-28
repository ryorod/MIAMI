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
import collections

from magenta.common import merge_hparams
from magenta.common import flatten_maybe_padded_sequences
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
import MusicVAE.lstm_models as lstm_models
from MusicVAE.base_model import MusicVAE
from MusicVAE.trained_model import TrainedModel
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

HParams = contrib_training.HParams
ds = tfp.distributions


class Config(collections.namedtuple(
    'Config',
    ['model', 'controller_model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

  def values(self):
    return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
  config_dict = config.values()
  config_dict.update(update_dict)
  return Config(**config_dict)


CONFIG_MAP = {}


# class TrainedModelEncoder(base_model.BaseEncoder):
#   """Encodes given data with pre-trained model."""

#   def __init__(self, original_encoder):
#     self._original_encoder = original_encoder

#   def build(self, model, hparams, is_training=True):
#     self._model = model

#   def encode_with_trained_model(self, sequence, sequence_length):
#     return self._model.encode_tensors(sequence, sequence_length)


# class TrainedModelDecoder(base_model.BaseDecoder):
#   """Decodes latent vectors with pre-trained model."""

#   def __init__(self, original_decoder):
#     self._original_decoder = original_decoder

#   def build(self, model, hparams, output_depth, is_training=True):
#     self._model = model

#   def reconstruction_loss(self, x_input, x_target, x_length, z=None,
#                           c_input=None):
#     pass

#   def sample(self, n, max_length=None, z=None, c_input=None):
#     pass


class ControllerVAE(MusicVAE):
  def __init__(self, encoder, decoder):
    class TrainedModelEncoder(encoder):
      """Encodes given data with pre-trained model."""

      def build(self, model, hparams, is_training=False):
        super().build(hparams, is_training)
        self._model = model

      def encode_with_trained_model(self, sequence, sequence_length):
        return self._model.encode_tensors(sequence, sequence_length)

    class TrainedModelDecoder(decoder):
      """Decodes latent vectors with pre-trained model."""

      def build(self, model, hparams, output_depth, is_training=False):
        super().build(hparams, output_depth, is_training)
        self._model = model
        self._hparams = hparams

      def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                              c_input=None):
        decode_results = self._model.decode_to_tensors(
                              z, self._hparams.max_seq_len,
                              c_input=c_input, return_full_results=True)
        flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
        flat_rnn_output = flatten_maybe_padded_sequences(
            decode_results.rnn_output, x_length)
        r_loss, metric_map = self._flat_reconstruction_loss(
            flat_x_target, flat_rnn_output)

        batch_size = int(x_input.shape[0])
        # Sum loss over sequences.
        cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
        r_losses = []
        for i in range(batch_size):
          b, e = cum_x_len[i], cum_x_len[i + 1]
          r_losses.append(tf.reduce_sum(r_loss[b:e]))
        r_loss = tf.stack(r_losses)

        return r_loss, metric_map, decode_results

      def sample(self, n, max_length=None, z=None, c_input=None):
        pass

    self._encoder = TrainedModelEncoder()
    self._decoder = TrainedModelDecoder()


  def build(self, config, hparams, checkpoint_dir_or_path, output_depth, is_training=False):
    tf.logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                    self.encoder.__class__.__name__,
                    self.decoder.__class__.__name__, hparams.values())
    self.global_step = tf.train.get_or_create_global_step()
    self._hparams = hparams

    trained_model = TrainedModel(
      config, batch_size=hparams.batch_size,
      checkpoint_dir_or_path=checkpoint_dir_or_path)

    self._encoder.build(trained_model, hparams, is_training)
    self._decoder.build(trained_model, hparams, output_depth, is_training)


  def encode_controller_vae(self, sequence, sequence_length, control_sequence=None):
    hparams = self.hparams
    controller_z_size = hparams.controller_z_size
    intermediate_size = hparams.intermediate_size

    sequence = tf.to_float(sequence)
    if control_sequence is not None:
      control_sequence = tf.to_float(control_sequence)
      sequence = tf.concat([sequence, control_sequence], axis=-1)

    encoder_output, _, _ = self.encoder.encode_with_trained_model(sequence, sequence_length)

    for layer, size in enumerate(intermediate_size, 1):
      encoder_output = tf.layers.dense(
        encoder_output,
        size,
        activation=tf.nn.leaky_relu,
        name='encoder/intermediate_%i' % layer,
        kernel_initializer=tf.keras.initializers.glorot_normal)

    mu = tf.layers.dense(
        encoder_output,
        controller_z_size,
        name='encoder/mu',
        kernel_initializer=tf.keras.initializers.glorot_normal)
    sigma = tf.layers.dense(
        encoder_output,
        controller_z_size,
        activation=tf.nn.softplus,
        name='encoder/sigma',
        kernel_initializer=tf.keras.initializers.glorot_normal)

    return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)


  def decode_controller_vae(self, z):
    hparams = self.hparams
    z_size = hparams.z_size
    intermediate_size = reversed(hparams.intermediate_size)
    decoder_output = z

    for layer, size in enumerate(intermediate_size, 1):
      decoder_output = tf.layers.dense(
        decoder_output,
        size,
        activation=tf.nn.leaky_relu,
        name='decoder/intermediate_%i' % layer,
        kernel_initializer=tf.keras.initializers.glorot_normal)

    decoder_output = tf.layers.dense(
      decoder_output,
      z_size,
      activation=tf.nn.leaky_relu,
      name='decoder/output',
      kernel_initializer=tf.keras.initializers.glorot_normal)


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
      q_z = self.encode_controller_vae(input_sequence, x_length, control_sequence)
      z = q_z.sample()
      z = self.decode_controller_vae(z)

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



CONFIG_MAP['cat-drums_2bar_small_3dim'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    controller_model=ControllerVAE(lstm_models.BidirectionalLstmEncoder,
                        lstm_models.CategoricalLstmDecoder),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            intermediate_size=[128],
            controller_z_size=3,
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