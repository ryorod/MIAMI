from magenta.models.music_vae import lstm_models
import tensorflow.compat.v1 as tf


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
      split_start_inputs = tf.split(start_inputs, self._output_depths, axis=-1)

    sample_results = []
    # bass only
    with tf.variable_scope('core_decoder_1'):
      sample_results.append(self._core_decoders[1].sample(
        n,
        max_length,
        z=z,
        c_input=c_input,
        temperature=temperature,
        start_inputs=split_start_inputs[1],
        **core_sampler_kwargs))

    sample_ids, decode_results = list(zip(*sample_results))
    return (tf.concat(sample_ids, axis=-1),
            self._merge_decode_results(decode_results))