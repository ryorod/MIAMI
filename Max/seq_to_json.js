autowatch = 1;
inlets = 3;
outlets = 1;

var id_path;
int_1 = 1;
int_2 = 1;
function msg_int(v) {
  switch (this.inlet) {
    case 1:
      int_1 = v;
      break;

    case 2:
      int_2 = v;
      break;

    default:
      break;
  }

  id_path = "live_set tracks " + int_1 + " clip_slots " + int_2;
}

id_path = "live_set tracks " + 1 + " clip_slots " + 1;
var clipApi;

function init() {
  clipApi = new LiveAPI(id_path + " clip");
  post("ClipAPI", clipApi)
}

function addNotes(pitch, velocity, startTime, endTime) {
  if (clipApi == null) throw "No clip";

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

function clearNotes() {
  // メロディ生成のの場合は16小節なので16を64に変更
  clipApi.call("remove_notes_extended", 0, 127, 0, 8);
  post("removed all notes on " + id_path);
}
