"""Copy into the experiment root to queue the middle student after the largest."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
output = root / "large-distilled-1.92m"
dependency = json.loads((output / "dependency.json").read_text())
status = {"model": output.name, "stage": "train", "pid": os.getpid(), "queued": time.time(), "dependency": dependency}

def save(**fields):
    status.update(fields)
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(output / "status.json")

try:
    save(state="waiting_dependency")
    while True:
        process = subprocess.run(["ps", "-p", str(dependency["pid"]), "-o", "command="], capture_output=True, text=True)
        if not process.stdout.strip():
            break
        if str(root / "run-largest.py") not in process.stdout:
            raise RuntimeError("Dependency PID no longer matches the largest-student wrapper")
        time.sleep(30)
    previous = json.loads((root / "status.json").read_text())
    if previous.get("state") != "complete" or previous.get("exit_code") != 0:
        raise RuntimeError("Largest-student training did not finish successfully")
    json.loads((root / "large-distilled-4.01m/training_result.json").read_text())
    if (output / "train.log").exists():
        raise RuntimeError("Middle training was already attempted; inspect before restarting")
    command = json.loads((output / "train-command.json").read_text())
    with (output / "train.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        save(state="running", started=time.time(), child_pid=child.pid)
        code = child.wait()
    if code or not (output / "training_result.json").exists():
        raise RuntimeError(f"Middle training failed with exit code {code}")
    json.loads((output / "training_result.json").read_text())
    save(state="complete", exit_code=code, finished=time.time())
except Exception as error:
    save(state="failed", error=str(error), finished=time.time())
    raise
