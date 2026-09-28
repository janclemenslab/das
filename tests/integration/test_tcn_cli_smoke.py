from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import das.cli as main


pytestmark = [pytest.mark.integration]


def _write_wav(path: Path, samples: np.ndarray, samplerate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), samplerate)


def _write_annotations(path: Path, rows: list[list[float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "name,start_seconds,stop_seconds",
                *[f"{name},{start_seconds},{stop_seconds}" for name, start_seconds, stop_seconds in rows],
            ]
        )
    )


def _signal(length: int, frequency_scale: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * frequency_scale, length, dtype=np.float32)
    return 0.25 * np.sin(t)


def _make_flat_audio_dir(root: Path) -> None:
    for idx in range(5):
        stem = f"clip_{idx}"
        _write_wav(root / f"{stem}.wav", _signal(256, frequency_scale=idx + 1))
        _write_annotations(
            root / f"{stem}_annotations.csv",
            [
                ["pulse", 0.032, 0.080],
                ["pulse", 0.160, 0.208],
            ],
        )


def test_tcn_cli_train_evaluate_predict_roundtrip(tmp_path: Path):
    dataset_dir = tmp_path / "audio"
    output_dir = tmp_path / "run"
    predictions_dir = tmp_path / "predictions"
    _make_flat_audio_dir(dataset_dir)

    checkpoint_path = main.cli_main(
        [
            "train",
            "--data-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size=2",
            "--num-time-steps=64",
            "--num-workers=0",
            "--frontend=raw",
            "--frontend-num-channels=1",
            "--encoder=tcn",
            "--decoder=linear",
            "--encoder-hidden-size=4",
            "--encoder-num-layers=1",
            "--encoder-dilations=[1,2]",
            "--encoder-kernel-size=3",
            "--encoder-dropout=0.0",
            "--cross-entropy-weight=1.0",
            "--learning-rate=0.01",
            "--accelerator=cpu",
            "--num-devices=1",
            "--num-epochs=1",
        ]
    )

    assert Path(checkpoint_path).exists()
    assert (output_dir / "logs").exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["das"]["predict"]["num_time_steps"] == 64
    assert checkpoint["das"]["predict"]["batch_size"] == 2

    written_files = main.cli_main(
        [
            "predict",
            "--data-dir",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(predictions_dir),
            "--evaluate",
            "--batch-size=2",
            "--num-workers=0",
            "--accelerator=cpu",
            "--num-devices=1",
        ]
    )

    assert len(written_files) == 5
    assert all(Path(path).exists() for path in written_files)
