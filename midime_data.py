from magenta.models.music_vae import data
import numpy as np


class BassConverter(data.TrioConverter):
  def from_tensors(self, samples, controls=None):
    output_sequences = []
    dim_ranges = np.cumsum(self._split_output_depths)
    for i, s in enumerate(samples):
      bass_ns = self._melody_converter.from_tensors(
          [s[:, dim_ranges[0]:dim_ranges[1]]])[0]

      for n in bass_ns.notes:
        n.instrument = 1
        n.program = data.ELECTRIC_BASS_PROGRAM

      output_sequences.append(bass_ns)
    return output_sequences
