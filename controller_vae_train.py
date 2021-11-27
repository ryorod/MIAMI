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

"""ControllerVAE generation script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import MusicVAE.configs as configs
from magenta.models.music_vae import data
from MusicVAE.trained_model import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_string(
    'master', '',
    'The TensorFlow master to use.')
flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of NoteSequence examples. Overrides the config.')
flags.DEFINE_string(
    'latent_vectors_path', './tmp/latent_vectors.npy',
    'Path to a npy file of latent vectors.')
flags.DEFINE_bool(
    'save_latent_vectors', True,
    'Whether to save latent vectors in a npy file.')
flags.DEFINE_bool(
    'use_saved_latent_vectors', False,
    'Whether to use already-saved latent vectors for training instead of encoding examples.')
flags.DEFINE_bool(
    'use_random_vectors', True,
    'Whether to use already-saved latent vectors for training instead of encoding examples.')
flags.DEFINE_bool(
    'embed_decode', False,
    'Whether to embed given data and decode it.')
flags.DEFINE_integer(
    'data_size', 256,
    '')
flags.DEFINE_integer(
    'data_num', 10000,
    '')
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
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""

  examples_path_summary = tf.summary.text(
      'examples_path', tf.constant(examples_path, name='examples_path'),
      collections=[])

  hparams_dict = hparams.values()

  # Create a markdown table from hparams.
  header = '| Key | Value |\n| :--- | :--- |\n'
  keys = sorted(hparams_dict.keys())
  lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
  hparams_table = header + '\n'.join(lines) + '\n'

  hparam_summary = tf.summary.text(
      'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
    writer.add_summary(examples_path_summary.eval())
    writer.add_summary(hparam_summary.eval())
    writer.close()


def _get_input_tensors(dataset, config):
  """Get input tensors from dataset."""
  batch_size = config.hparams.batch_size
  iterator = tf.data.make_one_shot_iterator(dataset)
  (input_sequence, output_sequence, control_sequence,
   sequence_length) = iterator.get_next()
  input_sequence.set_shape(
      [batch_size, None, config.data_converter.input_depth])
  output_sequence.set_shape(
      [batch_size, None, config.data_converter.output_depth])
  if not config.data_converter.control_depth:
    control_sequence = None
  else:
    control_sequence.set_shape(
        [batch_size, None, config.data_converter.control_depth])
  sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

  return {
      'input_sequence': input_sequence,
      'output_sequence': output_sequence,
      'control_sequence': control_sequence,
      'sequence_length': sequence_length
  }


def train_controller_vae(train_dir,
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
  tf.gfile.MakeDirs(train_dir)
  is_chief = (task == 0)
  if is_chief:
    _trial_summary(
        config.hparams, config.train_examples_path or config.tfds_name,
        train_dir)
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(
        num_ps_tasks, merge_devices=True)):

      model = config.model
      model.build(config.hparams,
                  config.data_converter.output_depth)

      optimizer = model.train(**_get_input_tensors(dataset_fn(), config))

      hooks = []
      if num_sync_workers:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            num_sync_workers)
        hooks.append(optimizer.make_session_run_hook(is_chief))

      grads, var_list = list(zip(*optimizer.compute_gradients(model.loss)))
      global_norm = tf.global_norm(grads)
      tf.summary.scalar('global_norm', global_norm)

      if config.hparams.clip_mode == 'value':
        g = config.hparams.grad_clip
        clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
      elif config.hparams.clip_mode == 'global_norm':
        clipped_grads = tf.cond(
            global_norm < config.hparams.grad_norm_clip_to_zero,
            lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
                grads, config.hparams.grad_clip, use_norm=global_norm)[0],
            lambda: [tf.zeros(tf.shape(g)) for g in grads])
      else:
        raise ValueError(
            'Unknown clip_mode: {}'.format(config.hparams.clip_mode))
      train_op = optimizer.apply_gradients(
          list(zip(clipped_grads, var_list)),
          global_step=model.global_step,
          name='train_step')

      logging_dict = {'global_step': model.global_step,
                      'loss': model.loss}

      hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
      if num_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

      scaffold = tf.train.Scaffold(
          saver=tf.train.Saver(
              max_to_keep=checkpoints_to_keep,
              keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
      tf_slim.training.train(
          train_op=train_op,
          logdir=train_dir,
          scaffold=scaffold,
          hooks=hooks,
          save_checkpoint_secs=60,
          master=master,
          is_chief=is_chief)


def evaluate(train_dir,
             eval_dir,
             config,
             dataset_fn,
             num_batches,
             master=''):
  """Evaluate the model repeatedly."""
  tf.gfile.MakeDirs(eval_dir)

  _trial_summary(
      config.hparams, config.eval_examples_path or config.tfds_name, eval_dir)
  with tf.Graph().as_default():
    model = config.model
    model.build(config.hparams,
                config.data_converter.output_depth)

    eval_op = model.eval(
        **_get_input_tensors(dataset_fn().take(num_batches), config))

    hooks = [
        tf_slim.evaluation.StopAfterNEvalsHook(num_batches),
        tf_slim.evaluation.SummaryAtEndHook(eval_dir)
    ]
    tf_slim.evaluation.evaluate_repeatedly(
        train_dir,
        eval_ops=eval_op,
        hooks=hooks,
        eval_interval_secs=60,
        master=master)


def get_input_note_sequences(config):
  examples_path = config.train_examples_path + '/*.mid'
  midi_paths = glob.glob(examples_path)
  
  def _check_extract_examples(input_ns, path):
    """Make sure each input returns exactly one example from the converter."""
    tensors = config.data_converter.to_tensors(input_ns).outputs
    if not tensors or len(tensors) > 1:
      print(
          'MusicVAE configs have very specific input requirements. Could not '
          'extract any valid inputs from `%s`.' % path)
      isValid = False
      return isValid
    else:
      isValid = True
      return isValid

  note_sequences = []
  for _, input_path in enumerate(midi_paths):
    input_midi = os.path.expanduser(input_path)
    input = note_seq.midi_file_to_note_sequence(input_midi)
    if _check_extract_examples(input, input_path):
      note_sequences.append(input)
    else:
      continue
  print('Number of Valid NoteSequences: %i' % len(note_sequences))

  return note_sequences


def encode_dataset(
          model,
          config):
  
  if FLAGS.use_saved_latent_vectors:
    path = os.path.expanduser(FLAGS.latent_vectors_path)
    _, ext = os.path.splitext(path)
    if not ext == '.npy':
      raise ValueError(
        'Filename must end with .npy.')
    z = np.load(path)
    return z

  else:
    examples_path = config.train_examples_path + '/*.mid'
    dataset_paths = glob.glob(examples_path)
  
    def _check_extract_examples(input_ns, path):
      """Make sure each input returns exactly one example from the converter."""
      tensors = config.data_converter.to_tensors(input_ns).outputs
      if not tensors or len(tensors) > 1:
        print(
            'MusicVAE configs have very specific input requirements. Could not '
            'extract any valid inputs from `%s`.' % path)
        isValid = False
        return isValid
      else:
        isValid = True
        return isValid

    logging.info(
        'Attempting to extract examples from input MIDIs using config `%s`...',
        FLAGS.config)

    dataset = []
    for _, input_path in enumerate(dataset_paths):
      input_midi = os.path.expanduser(input_path)
      input = note_seq.midi_file_to_note_sequence(input_midi)
      if _check_extract_examples(input, input_path):
        dataset.append(input)
      else:
        continue
    print('Number of Latent Vectors: %i' % len(dataset))

    z, _, _ = model.encode(dataset)
  
    if FLAGS.save_latent_vectors:
      path = os.path.expanduser(FLAGS.latent_vectors_path)
      _, ext = os.path.splitext(path)
      if not ext == '.npy':
        raise ValueError(
          'Filename must end with .npy.')
      
      np.save(path, z)

    return z


def load_model(config_map):

  if FLAGS.checkpoint_file is None:
    raise ValueError(
        '`--checkpoint_file` must be specified.')

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config name: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config.data_converter.max_tensors_per_item = None

  logging.info('Loading model...')
  checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
  model = TrainedModel(
      config, batch_size=config.hparams.batch_size,
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  return model


def train(config_map,
        model,
        tf_file_reader=tf.data.TFRecordDataset,
        file_reader=tf.python_io.tf_record_iterator):

  if not FLAGS.run_dir:
    raise ValueError('Invalid run directory: %s' % FLAGS.run_dir)
  run_dir = os.path.expanduser(FLAGS.run_dir)
  train_dir = os.path.join(run_dir, 'train')

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

  if is_training:
    train_controller_vae(
        train_dir,
        config=config,
        dataset_fn=dataset_fn,
        checkpoints_to_keep=FLAGS.checkpoints_to_keep,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        num_steps=FLAGS.num_steps,
        master=FLAGS.master,
        num_sync_workers=FLAGS.num_sync_workers,
        num_ps_tasks=FLAGS.num_ps_tasks,
        task=FLAGS.task)
  else:
    num_batches = FLAGS.eval_num_batches or data.count_examples(
        config.eval_examples_path,
        config.tfds_name,
        config.data_converter,
        file_reader) // config.hparams.batch_size
    eval_dir = os.path.join(run_dir, 'eval' + FLAGS.eval_dir_suffix)
    evaluate(
        train_dir,
        eval_dir,
        config=config,
        dataset_fn=dataset_fn,
        num_batches=num_batches,
        master=FLAGS.master)


# def embed_decode(cvae_model, z):
#   print(z)
#   zz = cvae_model.embed(z)
#   print(zz)
#   _z = cvae_model.decode(zz)
#   print(_z)


# def create_train_data(data_size=256, data_num=10000):
#   train_data = np.random.randn(data_num, data_size).astype(np.float32)
#   return train_data


# def train_with_random_vectors(train_data):
#   logdir = os.path.expanduser(FLAGS.logdir)

#   cvae_model = cvae.CompressionVAE(
#                 train_data,
#                 dim_latent=3,
#                 iaf_flow_length=5,
#                 batch_size=128,
#                 batch_size_test=128,
#                 logdir=logdir)
  
#   cvae_model.train()

#   return cvae_model


def run(config_map):
  if FLAGS.use_random_vectors:
    train_data = create_train_data(data_size=FLAGS.data_size, data_num=FLAGS.data_num)
    cvae_model = train_with_random_vectors(train_data)
    if FLAGS.embed_decode:
      test_data = create_train_data(data_size=FLAGS.data_size, data_num=10)
      embed_decode(cvae_model, test_data)
  else:
    model = load_model(config_map)
    cvae_model, z = train(config_map, model)
    if FLAGS.embed_decode:
      embed_decode(cvae_model, z)


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
