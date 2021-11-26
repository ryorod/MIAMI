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
from MusicVAE.trained_model import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf

import CompressionVAE.cvae.cvae as cvae

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
    'embed_decode', False,
    'Whether to embed given data and decode it.')
flags.DEFINE_string(
    'tfds_name', None,
    'TensorFlow Datasets dataset name to use. Overrides the config.')
flags.DEFINE_string(
    'logdir', 'temp',
    'Location for where to save the model and other related files.')
flags.DEFINE_integer(
    'num_steps', 200000,
    'Number of training steps or `None` for infinite.')


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
      config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  return model


def train(config_map,
        model):

  logdir = os.path.expanduser(FLAGS.logdir)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config_update_map = {}
  if FLAGS.examples_path:
    config_update_map['train_examples_path'] = os.path.expanduser(
        FLAGS.examples_path)
  if FLAGS.tfds_name:
    if FLAGS.examples_path:
      raise ValueError(
          'At most one of --examples_path and --tfds_name can be set.')
    config_update_map['tfds_name'] = FLAGS.tfds_name
    config_update_map['eval_examples_path'] = None
    config_update_map['train_examples_path'] = None
  config = configs.update_config(config, config_update_map)

  z = encode_dataset(
      model,
      config)
  
  cvae_model = cvae.CompressionVAE(
                z,
                dim_latent=3,
                iaf_flow_length=5,
                batch_size=config.hparams.batch_size,
                batch_size_test=config.hparams.batch_size,
                logdir=logdir)
  
  cvae_model.train()

  return cvae_model, z


def embed_decode(cvae_model, z):
  print(z)
  zz = cvae_model.embed(z)
  print(zz)
  _z = cvae_model.decode(zz)
  print(_z)


def run(config_map):
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
