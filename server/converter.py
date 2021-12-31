import os
import threading
from datetime import datetime, timedelta
from logging import warn
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import yaml
from note_seq import sequence_proto_to_midi_file
from note_seq.protobuf.music_pb2 import NoteSequence

from generator import create_note_seq

with open(os.path.join(os.path.dirname(__file__), "..", "config.yml"), 'r') as yml:
    CONFIG = yaml.safe_load(yml)


def float2timedelta(d: float):
    raise NotImplementedError


def default_func_on_output(z: np.ndarray) -> None:
    print(f"NoteSeqence generated: {z}")


def default_func_on_midi_output(path: str) -> None:
    print("midi_path", path)


class ReceivedNotesManager:
    """ this is a class for handling and buffering received notes for 
    transforming them to MIDI and sending them to generative models via callback
    functions.

    received data: `x y z velocity_x velocity_y velocity_z object_id`
    """

    def __init__(self, verbose=True):
        self.n_threshold_notes = 3
        self.window_sec = 2.0
        self.on_output_drums: Callable[[np.ndarray], None] = default_func_on_output
        self.on_output_mel: Callable[[np.ndarray], None] = default_func_on_output
        self.on_output_bass: Callable[[np.ndarray], None] = default_func_on_output
        # self.on_output_midi: Callable[[str], Any] = default_func_on_midi_output
        self.z = None
        self.verbose = verbose

    def update_start_point(self, d: datetime) -> None:
        raise NotImplementedError

    def receive(self, z: np.ndarray, mode: str) -> None:
        # z: 3dim float32 ndarray

        self.z = z

        if self.verbose:
            print("-" * 20, "\n")
        
        self.output(mode)

        if self.verbose:
            print(
                f"z: {z}"
            )
        return

    def clear(self):
        self.notes = []
        self.times = []
        self.durations = []
        self.velocities = []
        print("Stored notes cleared!")
        return

    def output(self, mode: str):
        if mode == 'drums':
            self.on_output_drums(self.z, mode)

        if mode == 'mel':
            self.on_output_mel(self.z, mode)

        if mode == 'bass':
            self.on_output_bass(self.z, mode)

        if mode == 'all':
            # self.on_output_drums(self.z, mode)
            # self.on_output_mel(self.z, mode)
            # self.on_output_bass(self.z, mode)
            raise NotImplementedError
