autowatch = 1;
inlets = 1;
outlets = 1;

id_path = "live_set tracks " + 1 + " clip_slots " + 1;
var clipApi;

function init() {
  clipApi = new LiveAPI(id_path + " clip");
  post("ClipAPI", clipApi)
}

function setClip(track, clip) {
  id_path = "live_set tracks " + (track - 1) + " clip_slots " + (clip - 1);
  clipApi = new LiveAPI(id_path + " clip");
}

function addNotes(pitch, velocity, startTime, endTime) {
  if (clipApi == null) throw "No clip";

  clipApi.call("fire");

  var note = {
    pitch: pitch,
    start_time: startTime,
    duration: endTime - startTime,
    velocity: velocity,
  };

  var noteArray = [note];

  var notesObject = {
    notes: noteArray,
  };
  var notesJson = JSON.stringify(notesObject);

  post("added new notes: " + pitch);
  clipApi.call("add_new_notes", notesJson);
}

function clearNotes(time_span) {
  // メロディ生成のの場合は16小節なので16を64に変更
  clipApi.call("remove_notes_extended", 0, 127, 0, time_span);
  post("removed all notes on " + id_path);
}

function isBass() {
  clipApi.set("loop_end", 32);
}
