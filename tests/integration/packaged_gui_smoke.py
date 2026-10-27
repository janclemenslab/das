"""Run GUI startup checks in a desktop session (or under xvfb-run on Linux)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def main():
    executable = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="packaged-gui-", dir=output))
    audio = work / "recording.wav"
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000), 16000)
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
                "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_PLATFORM", "QT_API"):
        env.pop(key, None)
    if sys.platform == "win32":
        windows = Path(os.environ["SystemRoot"])
        env["PATH"] = os.pathsep.join(map(str, (windows / "System32", windows, windows / "System32/Wbem")))
    else:
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    checks = []
    for name, args in [("no-arguments", []), ("open-recording", ["gui", str(audio), "--skip-dialog"])]:
        log_path = work / f"{name}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([str(executable), *args], cwd=work, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=30)
                raise AssertionError(f"{name}: GUI exited ({code}): {log_path.read_text(errors='replace')}")
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        text = log_path.read_text(errors="replace")
        for error in ("Traceback (most recent call last)", "Failed to execute script", "Failed to create context"):
            assert error not in text, text
        checks.append(dict(check=name, running_after_seconds=30, startup_crash=False))
        print(f"PASS: {name} stayed running without a startup error.", flush=True)
    result = dict(executable=str(executable), work=str(work), checks=checks, visual_interaction_verified=False)
    (output / "gui-results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
