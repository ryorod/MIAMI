import os
import sys
from datetime import datetime
from logging import warn, debug, info
from typing import Any, Optional

import numpy as np
import yaml
from note_seq.protobuf.music_pb2 import NoteSequence

from generator import MusicVAEModel
from osc import OSCSender

sys.path.append(os.path.dirname(__file__))
with open(os.path.join(os.path.dirname(__file__), "..", "config.yml"), 'r') as yml:
    CONFIG = yaml.safe_load(yml)


def sample_note_recorder(n: int, t: datetime, v: int, d: float):
    print(f"note: {n}")
    print(f"time: {t}")
    print(f"velocity: {v}")
    print(f"duration: {d}")


def on_output_midi_file_func(osc_sender: Optional[OSCSender] = None):
    """ it returns the function object that called when the MIDI file that
    packages sound inputs from the MR interface
    """
    def closure(midi_path: str):
        if osc_sender:
            print("midi_path", midi_path, "sent to osc")
            osc_sender.send("/midi_path", midi_path)
        else:
            print("midi_path", midi_path)  # for python-shell on Node for Max
    return closure


def on_output_note_sequence_func(vae: MusicVAEModel,
                                 osc_sender: Optional[OSCSender] = None):
    """ it returns the function object that called when the NoteSequence
    object that packages sound inputs from the MR interface
    """
    def closure(z: np.ndarray, mode: str):
        # z: 3dim float32 ndarray

        generated = vae.decode(z)

        if generated is not None:
            midi_path = vae.write_midi(generated)

            if osc_sender:
                print(f"midi_path_vae_{mode}", midi_path, "sent to osc")
                osc_sender.send(f"/midi_path_vae_{mode}", midi_path)
            else:
                # for python-shell on Node for Max
                print(f"midi_path_vae_{mode}", midi_path)
        else:
            warn("Failed to generate NoteSequence")
    return closure
