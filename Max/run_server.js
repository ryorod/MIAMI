const Max = require("max-api");
const path = require("path");
const fs = require("fs");
const yaml = require("js-yaml");
const { PythonShell } = require("python-shell");

const log = (msg) => {
  Max.post(`log ${msg}`);
  console.log("log", msg);
};

// log(`Nodejs version: ${process.version}`);
log(`Loaded the ${path.basename(__filename)} script`);

const root = path.resolve("../");
log(`Root path: ${root}`);

const pyScriptPath = root + "/server/";
log(`Python scripts on: ${pyScriptPath}`);

const configpath = root + "/config.yml";
const config = yaml.load(fs.readFileSync(configpath, "utf8"));
log(`loaded config on: ${configpath}`);

const options = {
  mode: "text",
  pythonOptions: ["-u"],
  pythonPath: config.python_path,
  scriptPath: pyScriptPath,
};

const pyshell = new PythonShell("main.py", options);

log(`Running the python script... ${pyScriptPath}main.py`);
pyshell.send("");

pyshell.on("message", function (data) {
  log(data);
  Max.outlet(data);
});

Max.addHandler("kill", function () {
  pyshell.kill();
  log("Stoped python server!");
});
