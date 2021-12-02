// Store map object with the key of note number and the value of object id
// written by Atsuya Kobayashi

/**
 * @param {number | array[number, string]} inlet
 * if inlet value is number then it returns the objectId value (string) extracted from data store
 * if inlet value is array of number and string then set the pair value of note number and objectID instead of returning value
 */
inlets = 1;
outlets = 1;

// init in load js file
/**
 * @var {String} noteToObjectIdMap : index would be note number
 */
var noteToObjectIdMap = new Array(128);
for (var i = 0; i < noteToObjectIdMap.length; i++) {
  noteToObjectIdMap[i] = undefined;
}

/**
 * @@param {number} noteNum Number (1 - 128)
 */
function searchObjectId(noteNum) {
  post("received note: " + noteNum + "\n");

  if (noteToObjectIdMap[noteNum] !== undefined) {
    return noteToObjectIdMap[noteNum];
  }
  var minDistanse = 10000;
  var currentIdx = undefined;
  for (var idx = 1; idx < noteToObjectIdMap.length; idx++) {
    if (noteToObjectIdMap[idx] !== undefined) {
      if (Math.abs(noteNum - idx) < minDistanse) {
        minDistanse = Math.abs(noteNum - idx);
        currentIdx = idx;
      }
    }
  }
  if (currentIdx === undefined) {
    return "";
  } else {
    return noteToObjectIdMap[currentIdx];
  }
}

/**
 * @@param {number} noteNum Number (1 - 128)
 * @@param {string} objectId e.g. Cube-1-1, Sphere-Effect-1-20
 */
function setObjectId(noteNum, objectId) {
  noteToObjectIdMap[noteNum] = objectId;
  post("note=" + noteNum + " bound to objectId=" + objectId + "\n");
}

function bang() {
  post("current store\n");
  for (var idx = 0; idx < noteToObjectIdMap.length; idx++) {
    post(idx + ": " + noteToObjectIdMap[idx] + "\n");
  }
}

function msg_int(noteNum) {
  post("received int " + noteNum + "\n");
  var extractedObjectId = searchObjectId(noteNum);
  outlet(0, extractedObjectId);
}

function list() {
  var [noteNum, objectId] = arrayfromargs(arguments);
  setObjectId(noteNum, objectId);
}

function clear() {
  seqArray.length = 0;
}
