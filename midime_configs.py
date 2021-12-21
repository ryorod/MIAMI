# Copyright 2019 The Magenta Authors.
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

# Modification copyright 2020 Bui Quoc Bao.
# Add Latent Constraint VAE model.
# Add Small VAE model.

"""Configurations for MusicVAE models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.configs import trio_16bar_converter

from midime_base_model import LCMusicVAE, SmallMusicVAE
from midime_lstm_models import BasslineLstmDecoder

HParams = contrib_training.HParams


class Config(collections.namedtuple(
        'Config', [
            'model', 'hparams', 'note_sequence_augmenter', 'data_converter',
            'train_examples_path', 'eval_examples_path', 'tfds_name', 'pretrained_path',
            'var_train_pattern', 'encoder_train', 'decoder_train'])):
    """Config class."""
    def values(self):
        """Return value as dictionary."""
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
    """Update config with new values."""
    config_dict = config.values()
    config_dict.update(update_dict)
    return Config(**config_dict)


CONFIG_MAP = dict()

CONFIG_MAP['lc-cat-mel_2bar_big'] = Config(
    model=LCMusicVAE(lstm_models.BidirectionalLstmEncoder(),
                     lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=2,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            encoded_z_size=8,
            enc_rnn_size=[2048],
            dec_rnn_size=[128, 128],
            free_bits=0,
            max_beta=0.5,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
    pretrained_path=None,
    var_train_pattern=['latent_encoder', 'decoder'],
    encoder_train=False,
    decoder_train=True
)

CONFIG_MAP['cat-mel_2bar_big_3dim'] = Config(
    model=SmallMusicVAE(lstm_models.BidirectionalLstmEncoder(),
                        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            encoded_z_size=3,
            latent_encoder_layers=[1024, 256, 64],
            latent_decoder_layers=[64, 256, 1024],
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            max_beta=0.5,
            beta_rate=0.99999,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
    pretrained_path=None,
    var_train_pattern=['latent'],
    encoder_train=False,
    decoder_train=False
)

CONFIG_MAP['bass_16bar_3dim'] = Config(
    model=SmallMusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            BasslineLstmDecoder(
                core_decoders=[
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder()],
                output_depths=[
                    90,  # melody
                    90,  # bass
                    512,  # drums
                ]),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=256,
            max_seq_len=256,
            z_size=512,
            encoded_z_size=3,
            latent_encoder_layers=[1024, 256, 64],
            latent_decoder_layers=[64, 256, 1024],
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=trio_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
    pretrained_path=None,
    var_train_pattern=['latent'],
    encoder_train=False,
    decoder_train=False
)

CONFIG_MAP['cat-drums_2bar_small_3dim'] = Config(
    model=SmallMusicVAE(lstm_models.BidirectionalLstmEncoder(),
                        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            encoded_z_size=3,
            latent_encoder_layers=[512, 128, 32],
            latent_decoder_layers=[32, 128, 512],
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
    pretrained_path=None,
    var_train_pattern=['latent'],
    encoder_train=False,
    decoder_train=False
)
