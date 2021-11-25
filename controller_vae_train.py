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

"""MusicVAE generation script."""

# TODO(adarob): Add support for models with conditioning.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import glob

import MusicVAE.configs as configs
from MusicVAE.trained_model import TrainedModel
from magenta.models.music_vae import data
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf

import CompressionVAE.cvae as cvae

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

# generate
# flags.DEFINE_string(
#     'run_dir', None,
#     'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
# flags.DEFINE_string(
#     'mode', 'sample',
#     'Generate mode (either `sample` or `interpolate`).')
# flags.DEFINE_string(
#     'input_midi_1', None,
#     'Path of start MIDI file for interpolation.')
# flags.DEFINE_string(
#     'input_midi_2', None,
#     'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

# train
flags.DEFINE_string(
    'master', '',
    'The TensorFlow master to use.')
flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of NoteSequence examples. Overrides the config.')
flags.DEFINE_string(
    'tfds_name', None,
    'TensorFlow Datasets dataset name to use. Overrides the config.')
flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
flags.DEFINE_integer(
    'num_steps', 200000,
    'Number of training steps or `None` for infinite.')
flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches to use during evaluation or `None` for all batches '
    'in the data source.')
flags.DEFINE_integer(
    'checkpoints_to_keep', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
flags.DEFINE_integer(
    'keep_checkpoint_every_n_hours', 1,
    'In addition to checkpoints_to_keep, keep a checkpoint every N hours.')
flags.DEFINE_string(
    'mode', 'train',
    'Which mode to use (`train` or `eval`).')
# flags.DEFINE_string(
#     'config', '',
#     'The name of the config to use.')
flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values to merge '
    'with those in the config.')
flags.DEFINE_bool(
    'cache_dataset', True,
    'Whether to cache the dataset in memory for improved training speed. May '
    'cause memory errors for very large datasets.')
flags.DEFINE_integer(
    'task', 0,
    'The task number for this worker.')
flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter server tasks.')
flags.DEFINE_integer(
    'num_sync_workers', 0,
    'The number of synchronized workers.')
flags.DEFINE_string(
    'eval_dir_suffix', '',
    'Suffix to add to eval output directory.')
# flags.DEFINE_string(
#     'log', 'INFO',
#     'The threshold for what messages will be logged: '
#     'DEBUG, INFO, WARN, ERROR, or FATAL.')


def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def encode_dataset(
          model,
          config,
          dataset_fn,
          checkpoints_to_keep=5,
          keep_checkpoint_every_n_hours=1,
          num_steps=None,
          master='',
          num_sync_workers=0,
          num_ps_tasks=0,
          task=0):
  """Train loop."""
  # tf.gfile.MakeDirs(train_dir)
  # is_chief = (task == 0)
  # if is_chief:
  #   _trial_summary(
  #       config.hparams, config.train_examples_path or config.tfds_name,
  #       train_dir)
  # with tf.Graph().as_default():
  #   with tf.device(tf.train.replica_device_setter(
  #       num_ps_tasks, merge_devices=True)):

  # encoded = model.encode(dataset_fn())
  # print(encoded)
  
  examples_path = config.train_examples_path + '/*.mid'
  dataset_paths = glob.glob(examples_path)

  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  
  def _check_extract_examples(input_ns, path, input_number):
    """Make sure each input returns exactly one example from the converter."""
    isValid = True
    tensors = config.data_converter.to_tensors(input_ns).outputs
    if not tensors:
      print(
          'MusicVAE configs have very specific input requirements. Could not '
          'extract any valid inputs from `%s`. Try another MIDI file.' % path)
      # sys.exit()
      isValid = False
      return isValid
    elif len(tensors) > 1:
      basename = os.path.join(
          FLAGS.output_dir,
          '%s_input%d-extractions_%s-*-of-%03d.mid' %
          (FLAGS.config, input_number, date_and_time, len(tensors)))
      for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
        note_seq.sequence_proto_to_midi_file(
            ns, basename.replace('*', '%03d' % i))
      print(
          '%d valid inputs extracted from `%s`. Outputting these potential '
          'inputs as `%s`. Call script again with one of these instead.' %
          (len(tensors), path, basename))
      # sys.exit()
      isValid = False
      return isValid
    else:
      return isValid

  logging.info(
      'Attempting to extract examples from input MIDIs using config `%s`...',
      FLAGS.config)

  dataset = []
  for i, input_path in enumerate(dataset_paths):
    input_midi = os.path.expanduser(input_path)
    input = note_seq.midi_file_to_note_sequence(input_midi)
    if _check_extract_examples(input, input_path, i+1):
      dataset.append(input)
    else:
      continue
  print('Number of Latent Vectors: %i' % len(dataset))

  z, _, _ = model.encode(dataset)
  # file = open('./tmp/latent_vectors.txt', 'w')
  # file.write(str(z))
  # file.close()
  
  return z

  # dataset = []
  # for file in os.listdir(examples_path):
  #   file_path = os.path.join(examples_path, file)
  #   if os.path.isfile(file_path):
  #     dataset.append(file_path)

      # hooks = []
      # if num_sync_workers:
      #   optimizer = tf.train.SyncReplicasOptimizer(
      #       optimizer,
      #       num_sync_workers)
      #   hooks.append(optimizer.make_session_run_hook(is_chief))

      # grads, var_list = list(zip(*optimizer.compute_gradients(model.loss)))
      # global_norm = tf.global_norm(grads)
      # tf.summary.scalar('global_norm', global_norm)

      # if config.hparams.clip_mode == 'value':
      #   g = config.hparams.grad_clip
      #   clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
      # elif config.hparams.clip_mode == 'global_norm':
      #   clipped_grads = tf.cond(
      #       global_norm < config.hparams.grad_norm_clip_to_zero,
      #       lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
      #           grads, config.hparams.grad_clip, use_norm=global_norm)[0],
      #       lambda: [tf.zeros(tf.shape(g)) for g in grads])
      # else:
      #   raise ValueError(
      #       'Unknown clip_mode: {}'.format(config.hparams.clip_mode))
      # train_op = optimizer.apply_gradients(
      #     list(zip(clipped_grads, var_list)),
      #     global_step=model.global_step,
      #     name='train_step')

      # logging_dict = {'global_step': model.global_step,
      #                 'loss': model.loss}

      # hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
      # if num_steps:
      #   hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

      # scaffold = tf.train.Scaffold(
      #     saver=tf.train.Saver(
      #         max_to_keep=checkpoints_to_keep,
      #         keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
      # tf_slim.training.train(
      #     train_op=train_op,
      #     logdir=train_dir,
      #     scaffold=scaffold,
      #     hooks=hooks,
      #     save_checkpoint_secs=60,
      #     master=master,
      #     is_chief=is_chief)


def load_model(config_map):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

  if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
    raise ValueError(
        'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
  # if FLAGS.output_dir is None:
  #   raise ValueError('`--output_dir` is required.')
  # tf.gfile.MakeDirs(FLAGS.output_dir)
  # if FLAGS.mode != 'sample' and FLAGS.mode != 'interpolate':
  #   raise ValueError('Invalid value for `--mode`: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config name: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config.data_converter.max_tensors_per_item = None

#   if FLAGS.mode == 'interpolate':
#     if FLAGS.input_midi_1 is None or FLAGS.input_midi_2 is None:
#       raise ValueError(
#           '`--input_midi_1` and `--input_midi_2` must be specified in '
#           '`interpolate` mode.')
#     input_midi_1 = os.path.expanduser(FLAGS.input_midi_1)
#     input_midi_2 = os.path.expanduser(FLAGS.input_midi_2)
#     if not os.path.exists(input_midi_1):
#       raise ValueError('Input MIDI 1 not found: %s' % FLAGS.input_midi_1)
#     if not os.path.exists(input_midi_2):
#       raise ValueError('Input MIDI 2 not found: %s' % FLAGS.input_midi_2)
#     input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
#     input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

#     def _check_extract_examples(input_ns, path, input_number):
#       """Make sure each input returns exactly one example from the converter."""
#       tensors = config.data_converter.to_tensors(input_ns).outputs
#       if not tensors:
#         print(
#             'MusicVAE configs have very specific input requirements. Could not '
#             'extract any valid inputs from `%s`. Try another MIDI file.' % path)
#         sys.exit()
#       elif len(tensors) > 1:
#         basename = os.path.join(
#             FLAGS.output_dir,
#             '%s_input%d-extractions_%s-*-of-%03d.mid' %
#             (FLAGS.config, input_number, date_and_time, len(tensors)))
#         for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
#           note_seq.sequence_proto_to_midi_file(
#               ns, basename.replace('*', '%03d' % i))
#         print(
#             '%d valid inputs extracted from `%s`. Outputting these potential '
#             'inputs as `%s`. Call script again with one of these instead.' %
#             (len(tensors), path, basename))
#         sys.exit()
#     logging.info(
#         'Attempting to extract examples from input MIDIs using config `%s`...',
#         FLAGS.config)
#     _check_extract_examples(input_1, FLAGS.input_midi_1, 1)
#     _check_extract_examples(input_2, FLAGS.input_midi_2, 2)

  logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
  model = TrainedModel(
      config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  return model

#   if FLAGS.mode == 'interpolate':
#     logging.info('Interpolating...')
#     _, mu, _ = model.encode([input_1, input_2])
#     z = np.array([
#         _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, FLAGS.num_outputs)])
#     results = model.decode(
#         length=config.hparams.max_seq_len,
#         z=z,
#         temperature=FLAGS.temperature)
#   elif FLAGS.mode == 'sample':
#     logging.info('Sampling...')
#     results = model.sample(
#         n=FLAGS.num_outputs,
#         length=config.hparams.max_seq_len,
#         temperature=FLAGS.temperature)

#   basename = os.path.join(
#       FLAGS.output_dir,
#       '%s_%s_%s-*-of-%03d.mid' %
#       (FLAGS.config, FLAGS.mode, date_and_time, FLAGS.num_outputs))
#   logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)
  # for i, ns in enumerate(results):
  #   note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

#   logging.info('Done.')


def train(config_map,
        model,
        tf_file_reader=tf.data.TFRecordDataset,
        file_reader=tf.python_io.tf_record_iterator):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.
    tf_file_reader: The tf.data.Dataset class to use for reading files.
    file_reader: The Python reader to use for reading files.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  # if not FLAGS.run_dir:
  #   raise ValueError('Invalid run directory: %s' % FLAGS.run_dir)
  # run_dir = os.path.expanduser(FLAGS.run_dir)
  # train_dir = os.path.join(run_dir, 'train')

  if FLAGS.mode not in ['train', 'eval']:
    raise ValueError('Invalid mode: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  if FLAGS.hparams:
    config.hparams.parse(FLAGS.hparams)
  config_update_map = {}
  if FLAGS.examples_path:
    config_update_map['%s_examples_path' % FLAGS.mode] = os.path.expanduser(
        FLAGS.examples_path)
  if FLAGS.tfds_name:
    if FLAGS.examples_path:
      raise ValueError(
          'At most one of --examples_path and --tfds_name can be set.')
    config_update_map['tfds_name'] = FLAGS.tfds_name
    config_update_map['eval_examples_path'] = None
    config_update_map['train_examples_path'] = None
  config = configs.update_config(config, config_update_map)
  if FLAGS.num_sync_workers:
    config.hparams.batch_size //= FLAGS.num_sync_workers

  if FLAGS.mode == 'train':
    is_training = True
  elif FLAGS.mode == 'eval':
    is_training = False
  else:
    raise ValueError('Invalid mode: {}'.format(FLAGS.mode))

  def dataset_fn():
    return data.get_dataset(
        config,
        tf_file_reader=tf_file_reader,
        is_training=is_training,
        cache_dataset=FLAGS.cache_dataset)

  z = encode_dataset(
      model,
      # train_dir,
      config=config,
      dataset_fn=dataset_fn,
      checkpoints_to_keep=FLAGS.checkpoints_to_keep,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_steps=FLAGS.num_steps,
      master=FLAGS.master,
      num_sync_workers=FLAGS.num_sync_workers,
      num_ps_tasks=FLAGS.num_ps_tasks,
      task=FLAGS.task)
  
  cvae_model = cvae.CompressionVAE(
                z,
                dim_latent=3,
                iaf_flow_length=5,
                batch_size=config.hparams.batch_size,
                batch_size_test=config.hparams.batch_size,
                tb_logging=True)
  
  cvae_model.train()


def run(config_map):
    model = load_model(config_map)
    train(config_map, model)


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
