import argparse
import math
import os
import sys
import threading
import warnings
from datetime import date, datetime
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from converter import ReceivedNotesManager

with open(os.path.join(os.path.dirname(__file__), "..", "config.yml"), 'r') as yml:
    CONFIG = yaml.safe_load(yml)


class OSCServer:
    def __init__(self, ip: str, port: int, timeout_seconds=10) -> None:

        self.address_z_all = "/z_all"
        self.address_z_mel = "/z_mel"
        self.address_z_bass = "/z_bass"
        self.address_z_drums = "/z_drums"

        self.ip = ip
        self.port = port
        # self.timeout = timeout_seconds

        self.data_manager: Optional[ReceivedNotesManager] = None
        # self.on_received: Optional[Callable[[
        #     int, datetime, int, float], Any]] = None
        self.server: Optional[BlockingOSCUDPServer] = None

        # self.bypass_sender: Optional[OSCSender] = None

    def parse_message(self, input_args: str) -> List[str]:
        expected_arg_length = 3
        args: List[str] = input_args.split(" ")
        assert len(args) == expected_arg_length, \
            f"length of input OSC message not matches, expected {expected_arg_length} but got {len(args)}"
        return args

    def _on_received(self, unused_addr, args, *values) -> None:

        # TODO: よく分からん value / arg の扱いを明確にしておく
        # print("value size: ", len(values))
        # for el in values:
        #     print('\t', el)

        # if self.bypass_sender:
        #     self.bypass_sender.send(self.notes_address, args)

        if self.data_manager:
            if callable(self.data_manager.receive):
                xyz = self.parse_message(args)
                xyz_ndarray = np.array(xyz, dtype=np.float32).reshape([1, 3])
                xyz_ndarray = np.dot(xyz_ndarray, 10)

                self.data_manager.receive(xyz_ndarray)

    def run(self, single_thread=False) -> None:
        self.dispatcher = Dispatcher()

        try:
            # register receive notes event callback
            self.dispatcher.map(self.address_z_all, self._on_received)

            self.server = BlockingOSCUDPServer(
                (self.ip, self.port), self.dispatcher)

        except OSError:
            # for nodejs python-shell
            print(f"address {self.ip} may not be correct")
            print(f"Port {self.port} may be used")
            os.system(f"lsof -i :{self.port}")
            return

        print(f"Serving on {self.server.server_address}")
        if single_thread:
            self.server.serve_forever()
        else:
            server_thread = threading.Thread(target=self.server.serve_forever)
            server_thread.start()

        ## Todo: timeout処理
        # time.sleep(20)
        # self.server.shutdown()

    def calc_velocity(self, v_x: float, v_y: float, v_z: float) -> int:
        """ v_x, v_y, v_z => velocity (60 - 127) """
        # TODO: ここ確認する
        v = 10 * math.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
        v = 60 + v
        v = int(min(v, 127))
        return v

    def calc_note(self, x: float, y: float, z: float) -> int:
        """ x, y, z => note (20 - 100) """
        # TODO: ここ確認する
        # scale -5 - 5 to 0 - 127
        return int((y + 5.0) * 8.0 + 20)

    def calc_duration(self, x: float, y: float, z: float) -> float:
        """ x, y, z => note (0 - 1.0) """
        # TODO: ここ確認する
        # scale -5 - 5 to 0 - 1.0
        return (z + 5.0) / 10.0

    def __del__(self):
        if self.server is not None:
            self.server.shutdown()


class OSCSender:
    def __init__(self, ip: str, port: int) -> None:
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send(self, path: str, msg: str) -> None:
        assert path[0] == "/", "given osc address path is incorrect"
        self.client.send_message(path, msg)

    def __del__(self):
        if self.client is not None:
            del self.client
