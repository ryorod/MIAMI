from magenta.models.music_vae import data


class BassConverter(data.TrioConverter):
  def from_tensors(self, samples):
    output_sequences = []
    for i, s in enumerate(samples):
      bass_ns = self._melody_converter.from_tensors(
          [s[:, :self._split_output_depths[1]]])[0]

      for n in bass_ns.notes:
        n.instrument = 1
        n.program = data.ELECTRIC_BASS_PROGRAM

      ns = bass_ns
      ns.total_time = bass_ns.total_time
      output_sequences.append(ns)
    return output_sequences