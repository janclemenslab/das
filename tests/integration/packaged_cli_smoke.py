"""Run with a packaged executable path and a directory for test outputs."""

from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import yaml


def main():
    executable = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="packaged-cli-", dir=output))
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

    def run(name, args, expected_code=0):
        result = subprocess.run([str(executable), *args], cwd=work, env=env,
                                capture_output=True, text=True, errors="replace", timeout=180)
        log = result.stdout + result.stderr
        (work / f"{name}.log").write_text(log)
        assert result.returncode == expected_code, (name, result.returncode, log[-6000:])
        checks.append({"check": name, "exit_code": result.returncode})
        print(f"PASS: {name}", flush=True)
        return result.stdout

    assert "CLI for DAS." in run("help", ["--help"])
    assert f"DAS {version('das')}" in run("version", ["version"])
    assert yaml.safe_load(run("train-config", ["train", "--config", "fly", "--print-config"]))["mode"] == "train"
    assert yaml.safe_load(run("predict-config", ["predict", "--print-config"]))["mode"] == "predict"
    assert "Launch the GUI." in run("gui-help", ["gui", "--help"])
    run("convert-legacy-help", ["convert-legacy", "--help"])
    run("invalid-command", ["unknown-command"], 2)

    audio = work / "audio"
    audio.mkdir()
    for index in range(5):
        stem = audio / f"clip_{index}"
        samples = 0.25 * np.sin(np.linspace(0, 2 * np.pi * (index + 1), 256, dtype=np.float32))
        sf.write(stem.with_suffix(".wav"), samples, 1000)
        stem.with_name(stem.name + "_annotations.csv").write_text(
            "name,start_seconds,stop_seconds\npulse,0.032,0.080\npulse,0.160,0.208\n"
        )
    config = dict(data_dir=str(audio), output_dir=str(work / "run"), batch_size=2, num_time_steps=64,
                  num_workers=0, frontend_type="raw", frontend_num_channels=1, encoder_type="tcn",
                  decoder_type="linear", encoder_hidden_size=4, encoder_num_layers=1,
                  encoder_dilations=[1, 2], encoder_kernel_size=3, encoder_dropout=0.0,
                  cross_entropy_weight=1.0, learning_rate=0.01, accelerator="cpu", num_devices=1, num_epochs=1)
    config_path = work / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run("train", ["train", "--config", str(config_path)])
    checkpoints = list((work / "run/checkpoints").glob("*.ckpt"))
    assert checkpoints, "Training did not produce a checkpoint"
    checkpoint = next((path for path in checkpoints if path.name != "last.ckpt"), checkpoints[0])
    predictions_dir = work / "predictions"
    run("predict", ["predict", "--data-dir", str(audio), "--checkpoint", str(checkpoint),
                    "--output-dir", str(predictions_dir), "--evaluate", "--batch-size", "2",
                    "--num-workers", "0", "--accelerator", "cpu", "--num-devices", "1"])
    predictions = sorted(predictions_dir.glob("*_annotations.csv"))
    assert len(predictions) == 5, predictions
    result = dict(executable=str(executable), work=str(work), checks=checks,
                  checkpoint=str(checkpoint), prediction_files=list(map(str, predictions)))
    (output / "cli-results.json").write_text(json.dumps(result, indent=2))
    print("PASS: packaged checkpoint and all five prediction files verified.", flush=True)


if __name__ == "__main__":
    main()
