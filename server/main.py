import argparse
import logging
from logging import info, debug
import os
import sys
from datetime import datetime
from typing import Optional

import yaml
from note_seq.protobuf.music_pb2 import NoteSequence

from converter import ReceivedNotesManager
from event_handlers import (continue_generation_func, on_output_midi_file_func,
                            on_output_note_sequence_func)
from generator import MusicVAEModel
from osc import OSCSender, OSCServer

sys.path.append(os.path.dirname(__file__))
with open(os.path.join(os.path.dirname(__file__), "..", "config.yml"), 'r') as yml:
    CONFIG = yaml.safe_load(yml)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("---send_address",
                        default=CONFIG["osc"]["send_address"],
                        help="The ip to listen on")
    parser.add_argument("---receive_address",
                        default=CONFIG["osc"]["receive_address"],
                        help="The ip to listen on")
    parser.add_argument("---send_port", type=int,
                        default=CONFIG["osc"]["send_port"],
                        help="The port to send on")
    parser.add_argument("---receive_port", type=int,
                        default=CONFIG["osc"]["receive_port"],
                        help="The port to receive on")
    parser.add_argument('--separate_mode', action='store_true',
                        help="if you do not run python process via M4L device")
    parser.add_argument('--verbose', action='store_true',
                        help="log level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.DEBUG if args.verbose else logging.INFO)

    print("=" * 40)
    info("Creating OSC server...")
    server = OSCServer(args.receive_address, args.receive_port)
    server.data_manager = ReceivedNotesManager()
    # TODO: hard coded send port num, change it
    server.bypass_sender = OSCSender(args.send_address, 6565)
    vae = MusicVAEModel(CONFIG["model_path_vae"])
    vae.load_model()

    if args.separate_mode:  # run mannually via shell
        sender = OSCSender(args.send_address, args.send_port)
        print(
            f"Results will be sent to {args.send_address}, port: {args.send_port}")
        server.data_manager.on_output_midi = on_output_midi_file_func(sender)
        server.data_manager.on_output = on_output_note_sequence_func(
            vae, osc_sender=sender)
        server.on_trigger_continuous_generate = continue_generation_func(
            vae, osc_sender=sender)
    else:  # run this from nodejs runtime on Max for Live Device
        server.data_manager.on_output_midi = on_output_midi_file_func()
        server.data_manager.on_output = on_output_note_sequence_func(
            vae)
        server.on_trigger_continuous_generate = continue_generation_func(
            vae)
    print("Starting server process...")
    server.run(single_thread=True)
