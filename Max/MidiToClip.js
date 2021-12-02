// Store map object with the key of note number and the value of object id
// written by Atsuya Kobayashi

function log() {
  for (var i = 0, len = arguments.length; i < len; i++) {
    var message = arguments[i];
    if (message && message.toString) {
      var s = message.toString();
      if (s.indexOf("[object ") >= 0) {
        s = JSON.stringify(message);
      }
      post(s);
    } else if (message === null) {
      post("<null>");
    } else {
      post(message);
    }
  }
  post("\n");
}

function ClipSlot(track, clipSlot) {
  this.path = "live_set tracks " + track + " clip_slots " + clipSlot;
  this.liveObject = new LiveAPI(this.path);
  post("Clip Slot Object created: ", this.liveObject, "\n");
}

function Clip(track, clipSlot) {
  liveObject = new LiveAPI(
    "live_set tracks " + track + " clip_slots " + clipSlot + " clip"
  );
  if (liveObject == undefined) {
    post("Livve Object not instantiated", "\n");
  } else {
    post("mode", liveObject.mode ? "follows path" : "follows object", "\n");
    post("id is", liveObject.id, "\n");
    post("path is", liveObject.path, "\n");
    post("children are", liveObject.children, "\n");
    post('getcount("devices")', liveObject.getcount("devices"), "\n");
    this.liveObject = liveObject;
  }
}

Clip.prototype.getLength = function () {
  return this.liveObject.get("length");
};

Clip.prototype._parseNoteData = function (data) {
  var notes = [];
  // data starts with "notes"/count and ends with "done" (which we ignore)
  for (var i = 2, len = data.length - 1; i < len; i += 6) {
    // and each note starts with "note" (which we ignore) and is 6 items in the list
    var note = new Note(
      data[i + 1],
      data[i + 2],
      data[i + 3],
      data[i + 4],
      data[i + 5]
    );
    notes.push(note);
  }
  return notes;
};

Clip.prototype.getSelectedNotes = function () {
  var data = this.liveObject.call("get_selected_notes");
  return this._parseNoteData(data);
};

Clip.prototype.getNotes = function (
  startTime,
  timeRange,
  startPitch,
  pitchRange
) {
  if (!startTime) startTime = 0;
  if (!timeRange) timeRange = this.getLength();
  if (!startPitch) startPitch = 0;
  if (!pitchRange) pitchRange = 128;

  var data = this.liveObject.call(
    "get_notes",
    startTime,
    startPitch,
    timeRange,
    pitchRange
  );
  return this._parseNoteData(data);
};

Clip.prototype._sendNotes = function (notes) {
  var liveObject = this.liveObject;
  liveObject.call("notes", notes.length);
  notes.forEach(function (note) {
    liveObject.call(
      "note",
      note.getPitch(),
      note.getStart(),
      note.getDuration(),
      note.getVelocity(),
      note.getMuted()
    );
  });
  liveObject.call("done");
};

Clip.prototype.replaceSelectedNotes = function (notes) {
  this.liveObject.call("select_all_notes");
  this.liveObject.call("replace_selected_notes");
  this._sendNotes(notes);
};

Clip.prototype.setNotes = function (notes) {
  this.liveObject.call("set_notes");
  this._sendNotes(notes);
};

//------------------------------------------------------------------------------
// Note class

function Note(pitch, start, duration, velocity, muted) {
  this.pitch = pitch;
  this.start = start;
  this.duration = duration;
  this.velocity = velocity;
  this.muted = muted;
}

Note.prototype.toString = function () {
  return (
    "{pitch:" +
    this.pitch +
    ", start:" +
    this.start +
    ", duration:" +
    this.duration +
    ", velocity:" +
    this.velocity +
    ", muted:" +
    this.muted +
    "}"
  );
};

Note.MIN_DURATION = 1 / 128;

Note.prototype.getPitch = function () {
  if (this.pitch < 0) return 0;
  if (this.pitch > 127) return 127;
  return this.pitch;
};

Note.prototype.getStart = function () {
  // we convert to strings with decimals to work around a bug in Max
  // otherwise we get an invalid syntax error when trying to set notes
  if (this.start <= 0) return "0.0";
  return this.start.toFixed(4);
};

Note.prototype.getDuration = function () {
  if (this.duration <= Note.MIN_DURATION) return Note.MIN_DURATION;
  return this.duration.toFixed(4); // workaround similar bug as with getStart()
};

Note.prototype.getVelocity = function () {
  if (this.velocity < 0) return 0;
  if (this.velocity > 127) return 127;
  return this.velocity;
};

Note.prototype.getMuted = function () {
  if (this.muted) return 1;
  return 0;
};

function humanize(type, maxTimeDelta, maxVelocityDelta) {
  var humanizeVelocity = false,
    humanizeTime = false;

  switch (type) {
    case "velocity":
      humanizeVelocity = true;
      break;
    case "time":
      humanizeTime = true;
      break;
    default:
      humanizeVelocity = humanizeTime = true;
  }

  if (!maxTimeDelta) maxTimeDelta = 0.05;
  if (!maxVelocityDelta) maxVelocityDelta = 5;

  clip = new Clip();
  notes = clip.getSelectedNotes();
  notes.forEach(function (note) {
    if (humanizeTime) note.start += maxTimeDelta * (2 * Math.random() - 1);
    if (humanizeVelocity)
      note.velocity += maxVelocityDelta * (2 * Math.random() - 1);
  });
  clip.replaceSelectedNotes(notes);
}

//------------------------------------------------------------------------------

inlets = 2;
outlets = 1;

var targetSlot = null;
var track = 0;
var clipSlot = 0;

var clip = null;
var notes = [];

var lastNoteStartTime = new Array(128);
var lastNoteVelocity = new Array(128);
for (var i = 0; i < lastNoteStartTime.length; i++) {
  lastNoteStartTime[i] = undefined;
  lastNoteVelocity[i] = undefined;
}

function createClip(data) {
  if (!targetSlot) {
    targetSlot = new ClipSlot(track, clipSlot);
  }
  if (!clip) {
    targetSlot.liveObject.call("delete_clip");
    targetSlot.liveObject.call("create_clip", 16);
    clip = new Clip(track, clipSlot);
    post("Clip initialized", clip.liveObject.path, "\n");
  }

  var [time, note, velocity, isMuted] = data;

  if (lastNoteStartTime[Number(note)] == undefined) {
    // note on
    lastNoteStartTime[Number(note)] = time / 1000;
    lastNoteVelocity[Number(note)] = velocity;
  } else {
    // note off
    var duration = (time - lastNoteStartTime[Number(note)]) / 1000;
    if (clip) {
      // pitch, start, duration, velocity, muted
      post(
        [
          note,
          lastNoteStartTime[Number(note)],
          duration,
          lastNoteVelocity[Number(note)],
          isMuted,
        ].join(", ") + "\n"
      );

      notes.push(
        new Note(
          note,
          lastNoteStartTime[Number(note)],
          duration,
          lastNoteVelocity[Number(note)],
          isMuted
        )
      );
    }
    lastNoteStartTime[note] = undefined;
    lastNoteVelocity[note] = undefined;
  }
}

function init(slots) {
  [track, clipSlot] = slots;
  targetSlot = new ClipSlot(track, clipSlot);
}

function list() {
  if (inlet == 0) {
    createClip(arrayfromargs(arguments));
  } else if (inlet == 1) {
    init(arrayfromargs(arguments));
  }
}

function finish(arg) {
  if (clip) {
    clip.setNotes(notes);
    // clip.replaceSelectedNotes(notes);
    post("clip set: ", notes.length, " notes", "\n");
    clip.liveObject.call("fire");
    post("clip fired", "\n");
  }
  clip = null;
  notes = [];
}

function bang() {}

function info() {
  post("track: ", track, "\n");
  post("clip slot: ", clipSlot, "\n");
  post("slot object: ", targetSlot && targetSlot.liveObject, "\n");
  post("clip object: ", clip && clip.liveObject, "\n");
  post("notes: ", notes, "\n");
}

function createNewClip() {
  targetSlot && targetSlot.liveObject.call("create_clip", 16);
}

function deleteCurrentClip() {
  targetSlot && targetSlot.liveObject.call("delete_clip");
}
