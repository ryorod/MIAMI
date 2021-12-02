from functools import wraps
import os
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from logging import warn, debug, info
from typing import Any, List, Optional

import numpy as np
import yaml
from magenta.models.music_vae.trained_model import TrainedModel
from magenta.models.shared import sequence_generator_bundle
from note_seq import sequence_proto_to_midi_file
from note_seq.protobuf import generator_pb2, music_pb2
from note_seq.protobuf.music_pb2 import NoteSequence

warnings.simplefilter('ignore')

with open(os.path.join(os.path.dirname(__file__),
                       "..", "config.yml"), 'r') as yml:
    CONFIG = yaml.safe_load(yml)

import sys
sys.path.append('../')
from ControllerVAE import controller_vae_configs as configs


def create_note_seq(notes: List[int],
                    times: List[datetime],
                    durations: List[float],
                    velocities: List[int],
                    is_drum=False,
                    verbose=False) -> NoteSequence:
    """ returns `music_pb2.NoteSequence` instance from MIDI info """
    # pyright: reportGeneralTypeIssues=false
    sequence = NoteSequence()
    for n, t, d, v in zip(notes, times, durations, velocities):

        relative_time = (t - times[0]).total_seconds()
        # pylint: disable=maybe-no-member
        sequence.notes.add(pitch=n, start_time=relative_time,
                           end_time=relative_time + d,
                           velocity=v,
                           is_drum=is_drum,
                           # https://ja.wikipedia.org/wiki/General_MIDI
                           instrument=5)

    sequence.total_time = (times[-1] - times[0]).total_seconds()

    # TODO: handle tempo data on sequence object
    return sequence


class ModelInterface(metaclass=ABCMeta):

    model: Optional[Any]
    model_path: str
    midi_output_dir: str
    previous_sequence: Optional[NoteSequence]
    previous_sequence_updated: float  # updated timestamp

    @abstractmethod
    def load_model(self, path: Optional[str]):
        pass

    @abstractmethod
    def generate_from_sequence(self):
        pass

    @abstractmethod
    def write_midi(self):
        pass


class MusicVAEModel(ModelInterface):

    """ Magenta's MusicVAE trained model wrapper class """

    def __init__(self, model_path: str,
                 config_map_key: Optional[str] = "cat-drums_2bar_small_3dim",
                 midi_output_dir: str = CONFIG["midi_output_dir"]) -> None:
        self.latest_z: Optional[np.ndarray] = None
        self.model: Optional[TrainedModel] = None
        self.model_path = model_path
        self.midi_output_dir = midi_output_dir
        self.config_map_key = config_map_key
        self.max_seq_len = configs.CONFIG_MAP[config_map_key].hparams.max_seq_len

        # TODO: 長さ決めておく
        self.previous_sequence: Optional[NoteSequence] = None
        self.previous_length: int = 32
        self.previous_sequence_updated: Optional[float] = None

    def load_model(self, path: Optional[str] = None) -> None:
        # This will download the mel_2bar_big checkpoint. There are more checkpo
        # ints that youcan use with this model, depending on what kind of output
        # you want
        # See the list of checkpoints: https://github.com/magenta/magenta/tree/m
        # aster/magenta/models/music_vae#pre-trained-checkpoints
        # !gsutil - q - m cp - R gs: // download.magenta.tensorflow.org/models/m
        # sic_vae/colab2/checkpoints/mel_2bar_big.ckpt.* / content/
        info("Initializing Music VAE...")
        try:
            self.model = TrainedModel(
                configs.CONFIG_MAP[self.config_map_key],
                batch_size=4,
                checkpoint_dir_or_path=path if path else self.model_path)
            info(f"Loding Model Done!: {self.model_path}")
        except Exception as e:
            warn(f"Failed to load model: {self.model_path}")
            warn(e)

    def get_configs(self):
        print(configs)
        return configs

    def move_z(self) -> np.ndarray:
        # TODO 実装する
        raise NotImplementedError

    def encode(self, sequence: NoteSequence) -> Optional[np.ndarray]:
        """NoteSequenceをEncodeしてｚにする
        ref: https://colab.research.google.com/notebooks/magenta/hello_magenta/
        hello_magenta.ipynb#scrollTo=QhtRBNNf05CA
        """
        if self.model:
            try:
                z, _, _ = self.model.encode([sequence])
                self.latest_z = np.array(z)
                return self.latest_z
            except Exception as e:
                warn(f"Failed to encode sequence: {e}")
        else:
            warn("MelodyRNN model not loaded!, call `MusicVAEModel.load_model()`")

    def decode(self, z: np.ndarray,
               length: Optional[int] = 32) -> Optional[NoteSequence]:
        """
        zからNoteSequenceをdecodeする関数

        ref: https://colab.research.google.com/notebooks/magenta/hello_magenta/
        hello_magenta.ipynb#scrollTo=QhtRBNNf05CA
        """
        if self.model:
            length = self.max_seq_len
            decoded = self.model.decode(z, length=length)
            info(f"decoded {len(decoded)} NoteSequence objects")
            self.previous_sequences = decoded
            return decoded[0]
        else:
            warn("MelodyRNN model not loaded!, call `MusicVAEModel.load_model()`")

    def write_midi(self, notes) -> str:
        midi_file_name = datetime.now().strftime("%y%m%d_%H%M%S")
        midi_path = os.path.join(
            self.midi_output_dir, "music_vae", f"{midi_file_name}.mid")
        try:
            sequence_proto_to_midi_file(notes, midi_path)
            return midi_path
        except Exception as e:
            warn(f"Failed to generate and write midi file to: {midi_path}", e)
            return ""

    def update_previous_sequence(self, sequence: NoteSequence) -> None:
        self.previous_sequence = sequence
        self.previous_sequence_updated = datetime.now().timestamp()

    def generate_from_sequence(self, input: NoteSequence, length=64,
                               noise_bias=-4) -> Optional[NoteSequence]:
        z = self.encode(input)
        if z is None:
            warn("failed to get encoded z from input sequence")
            return None
        noise = self.get_noise(z, noise_bias)
        z_dash = z + noise
        output = self.decode(z_dash)
        self.update_previous_sequence(output)
        return output

    def interpolate_from_sequence(self, input: NoteSequence,
                                  noise_bias=-4) -> Optional[NoteSequence]:
        if self.previous_sequence is None:
            self.previous_sequence = input
        if self.model:
            sequence = self.model.interpolate(self.previous_sequence,
                                              input, num_steps=1,
                                              length=self.previous_length)[0]
            self.latest_z = self.encode(sequence)
            self.update_previous_sequence(sequence)
            return sequence

    def generate_continuous(self, noise_bias=-4,
                            interporate=False) -> Optional[NoteSequence]:

        def get_decorded_output() -> Optional[NoteSequence]:
            if self.latest_z:
                noise = self.get_noise(self.latest_z, noise_bias)
                z_dash = self.latest_z + noise
                output = self.decode(z_dash)
                self.update_previous_sequence(output)
                if output and interporate:
                    return self.interpolate_from_sequence(output)
                return output
            else:
                warn("Failed to update `latest_z` by encoding Sequence")
                return

        if self.latest_z is not None:
            return get_decorded_output()
        elif self.previous_sequence is not None:
            self.encode(self.previous_sequence)
            return get_decorded_output()
        else:
            warn("previous generated note data not found")
            return

    def get_noise(self, z: np.ndarray, noise_bias: float = -4.0) -> np.ndarray:
        return np.random.randn(*z.shape) * 10**(noise_bias)


if __name__ == "__main__":
    pass
