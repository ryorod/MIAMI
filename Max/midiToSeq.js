// MIDI Clip Importer
// written by Keisuke Nohara

inlets = 1;
outlets = 2;

var seqArray = new Array();

function list() {
  value = arguments;
  seqArray.push([value[0], value[1], value[2], value[3]]);
}

function scaleToInt(i) {
  if (i < 0) {
    return 0;
  } else if (i > 127) {
    return 127;
  }
  return Math.floor(i);
}

function bang() {
  outlet(0, "call set_notes");
  outlet(0, "call notes " + seqArray.length);

  for (var i = 0; i < seqArray.length; i++) {
    // pitch, start, duration, velocity, muted
    outlet(
      0,
      "call note " +
        seqArray[i][1] +
        " " +
        seqArray[i][0].toFixed(6) +
        " " +
        Math.max(0.05, seqArray[i][3]) +
        " " +
        scaleToInt(seqArray[i][2]) + // velocity
        " " +
        0 // muted
    );
  }
  outlet(1, bang);
}

function clear() {
  seqArray.length = 0;
}
