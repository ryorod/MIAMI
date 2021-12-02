autowatch = 1;
inlets = 1;
outlets = 1;

var pitchArray = new Array();
var velocityArray = new Array();
var noteOnArray = new Array();
var durationArray = new Array();

var noteArray = new Array();

id_path = "live_set tracks " + 0 + " clip_slots " + 0;
var clipApi = new LiveAPI(id_path + " clip");

function stockNotes(pitch, velocity, startTime, endTime) {
  pitchArray.push(pitch);
  velocityArray.push(velocity);
  noteOnArray.push(startTime);
  durationArray.push(endTime - startTime);
}

function addNotes() {
  if (clipApi == null) throw "No clip";

  for (i = 0; i < pitchArray.length; i++) {
    var note = {
      pitch: pitchArray[i],
      start_time: noteOnArray[i],
      duration: durationArray[i],
      velocity: velocityArray[i],
    };
    noteArray.push(note);
  }

  var notesObject = {
    notes: noteArray,
  };
  var notesJson = JSON.stringify(notesObject);

  clipApi.call("add_new_notes", notesJson);
}

function clear() {
  clipApi.call("remove_notes_extended", 0, 127, 0, 16);

  noteArray.length = 0;
  noteOnArray.length = 0;
  durationArray.length = 0;
  pitchArray.length = 0;
  velocityArray.length = 0;
}
