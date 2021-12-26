from magenta.models.music_vae import lstm_models
from magenta.models.music_vae import lstm_utils
import tensorflow.compat.v1 as tf
import numpy as np


class BasslineLstmDecoder(lstm_models.SplitMultiOutLstmDecoder):
  def build(self, hparams, output_depth, is_training=True):
    if sum(self._output_depths) != output_depth:
      raise ValueError(
          'Decoder output depth does not match sum of sub-decoders: %s vs %d' %
          (self._output_depths, output_depth))
    self.hparams = hparams
    self._is_training = is_training

    # bass only
    with tf.variable_scope('core_decoder_1'):
      self._core_decoders[1].build(hparams, self._output_depths[1], is_training)


  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             start_inputs=None, **core_sampler_kwargs):
    if z is not None and int(z.shape[0]) != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0], n))

    if max_length is None:
      # TODO(adarob): Support variable length outputs.
      raise ValueError(
          'SplitMultiOutLstmDecoder requires `max_length` be provided during '
          'sampling.')

    if start_inputs is None:
      split_start_inputs = [None] * len(self._output_depths)
    else:
      split_start_inputs = tf.split(start_inputs, self._output_depths[1], axis=-1)

    sample_results = []
    # bass only
    with tf.variable_scope('core_decoder_1'):
      sample_results.append(self._core_decoders[1].sample(
        n,
        max_length,
        z=z,
        c_input=c_input,
        temperature=temperature,
        start_inputs=split_start_inputs[0],
        **core_sampler_kwargs))

    sample_ids, decode_results = list(zip(*sample_results))
    return (tf.concat(sample_ids, axis=-1),
            self._merge_decode_results(decode_results))

class BasslineHierarchicalLstmDecoder(lstm_models.HierarchicalLstmDecoder):
  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.
    Args:
      x_input: Batch of decoder input sequences of concatenated segmeents for
        teacher forcing, sized `[batch_size, max_seq_len, output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max_seq_len, output_depth]`.
      x_length: Length of input/output sequences, sized
        `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
        each length must either equal `max_seq_len` or 0. In this case, the
        segment lengths are assumed to be constant and the total length will be
        evenly divided amongst the segments.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
        `[batch_size, max_seq_len, control_depth]`. Required if conditioning on
        control sequences.
    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.
    Raises:
      ValueError: If `c_input` is provided in re-encoder mode.
    """
    if self._hierarchical_encoder and c_input is not None:
      raise ValueError(
          'Re-encoder mode unsupported when conditioning on controls.')

    batch_size = int(x_input.shape[0])

    x_length = lstm_utils.maybe_split_sequence_lengths(
        x_length, np.prod(self._level_lengths[:-1]), self._total_length)

    hier_input = self._reshape_to_hierarchy(x_input)
    hier_target = self._reshape_to_hierarchy(x_target)
    hier_length = self._reshape_to_hierarchy(x_length)
    hier_control = (
        self._reshape_to_hierarchy(c_input) if c_input is not None else None)

    loss_outputs = []

    def base_train_fn(embedding, hier_index):
      """Base function for training hierarchical decoder."""
      split_size = self._level_lengths[-1]
      split_input = hier_input[hier_index]
      split_target = hier_target[hier_index]
      split_length = hier_length[hier_index]
      split_control = (
          hier_control[hier_index] if hier_control is not None else None)

      with tf.variable_scope('core_decoder_1'):
        res = self._core_decoder.reconstruction_loss(
            split_input, split_target, split_length, embedding, split_control)
        loss_outputs.append(res)
        decode_results = res[-1]

      if self._hierarchical_encoder:
        # Get the approximate "sample" from the model.
        # Start with the inputs the RNN saw (excluding the start token).
        samples = decode_results.rnn_input[:, 1:]
        # Pad to be the max length.
        samples = tf.pad(
            samples,
            [(0, 0), (0, split_size - tf.shape(samples)[1]), (0, 0)])
        samples.set_shape([batch_size, split_size, self._output_depth])
        # Set the final value based on the target, since the scheduled sampling
        # helper does not sample the final value.
        samples = lstm_utils.set_final(
            samples,
            split_length,
            lstm_utils.get_final(split_target, split_length, time_major=False),
            time_major=False)
        # Return the re-encoded sample.
        return self._hierarchical_encoder.level(0).encode(
            sequence=samples,
            sequence_length=split_length)
      elif self._disable_autoregression:
        return None
      else:
        return tf.concat(tf.nest.flatten(decode_results.final_state), axis=-1)

    z = tf.zeros([batch_size, 0]) if z is None else z
    self._hierarchical_decode(z, base_train_fn)

    # Accumulate the split sequence losses.
    r_losses, metric_maps, decode_results = list(zip(*loss_outputs))

    # Merge the metric maps by passing through renamed values and taking the
    # mean across the splits.
    merged_metric_map = {}
    for metric_name in metric_maps[0]:
      metric_values = []
      for i, m in enumerate(metric_maps):
        merged_metric_map['segment/%03d/%s' % (i, metric_name)] = m[metric_name]
        metric_values.append(m[metric_name][0])
      merged_metric_map[metric_name] = (
          tf.reduce_mean(metric_values), tf.no_op())

    return (tf.reduce_sum(r_losses, axis=0),
            merged_metric_map,
            self._merge_decode_results(decode_results))