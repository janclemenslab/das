from pathlib import Path

import numpy as np
import pytest
import torch

from das.data import NPYDirDataModule, is_npy_dir, load_npy_dir_attrs


def _write_npy_dir_fixture(
    tmp_path: Path,
    name: str = "legacy.npy",
    samplerate: int = 16_000,
    channels: int = 1,
) -> Path:
    root = tmp_path / name
    root.mkdir()
    np.save(
        root / "attrs.npy",
        {
            "class_names": ["noise", "song"],
            "class_types": ["segment", "segment"],
            "samplerate_x_Hz": samplerate,
            "samplerate_y_Hz": samplerate,
        },
        allow_pickle=True,
    )

    for split, samples in {"train": 128, "val": 64, "test": 64}.items():
        split_dir = root / split
        split_dir.mkdir()
        if channels == 1:
            x = np.zeros((samples,), dtype=np.float16)
            x[8:16] = 0.5
        else:
            x = np.zeros((samples, channels), dtype=np.float16)
            x[8:16, 0] = 0.5
        y = np.zeros((samples, 2), dtype=np.float16)
        y[:, 0] = 1.0
        y[24:40, 0] = 0.0
        y[24:40, 1] = 1.0
        np.save(split_dir / "x.npy", x)
        np.save(split_dir / "y.npy", y)

    return root


def test_npy_dir_datamodule_reads_attrs_and_returns_waveform_batches(tmp_path: Path):
    data_dir = _write_npy_dir_fixture(tmp_path)

    assert is_npy_dir(data_dir)
    assert load_npy_dir_attrs(data_dir)["samplerate_x_Hz"] == 16_000

    datamodule = NPYDirDataModule(
        data_dir=str(data_dir),
        batch_size=2,
        num_time_steps=32,
        chunk_stride=16,
        hop_s=4 / 16_000,
        num_workers=0,
        persistent_workers=False,
    )
    batch = next(iter(datamodule.train_dataloader()))
    inputs, input_lengths, targets, target_lengths = batch

    assert datamodule.class_names == ["noise", "song"]
    assert datamodule.num_classes == 2
    assert datamodule.samplerate_x_hz == 16_000
    assert datamodule.samplerate_y_hz == 16_000
    assert datamodule.input_num_channels == 1
    assert datamodule.has_val is True
    assert datamodule.has_test is True
    assert inputs.shape == (2, 32)
    assert torch.equal(input_lengths, torch.tensor([32, 32]))
    assert targets.shape == (2, 9, 2)
    assert torch.equal(target_lengths, torch.tensor([9, 9]))


def test_npy_dir_datamodule_preserves_multichannel_waveforms(tmp_path: Path):
    data_dir = _write_npy_dir_fixture(tmp_path, channels=2)

    datamodule = NPYDirDataModule(
        data_dir=str(data_dir),
        batch_size=2,
        num_time_steps=32,
        chunk_stride=16,
        hop_s=1 / 16_000,
        num_workers=0,
        persistent_workers=False,
    )
    inputs, input_lengths, targets, target_lengths = next(iter(datamodule.train_dataloader()))

    assert datamodule.input_num_channels == 2
    assert inputs.shape == (2, 2, 32)
    assert torch.equal(input_lengths, torch.tensor([32, 32]))
    assert targets.shape == (2, 32, 2)
    assert torch.equal(target_lengths, torch.tensor([32, 32]))


def test_npy_dir_split_stats_report_files_and_minutes(tmp_path: Path):
    data_dir = _write_npy_dir_fixture(tmp_path)

    datamodule = NPYDirDataModule(
        data_dir=str(data_dir),
        batch_size=2,
        num_time_steps=32,
        chunk_stride=16,
        hop_s=4 / 16_000,
        num_workers=0,
        persistent_workers=False,
    )
    stats = {row["split"]: row for row in datamodule.split_stats()}

    assert stats["train"]["audio_file_count"] == 1
    assert stats["train"]["audio_minutes"] == pytest.approx(128 / 16_000 / 60)
    assert stats["train"]["annotation_file_count"] == 1
    assert stats["train"]["annotation_minutes"] == pytest.approx(128 / 16_000 / 60)
    assert stats["val"]["audio_minutes"] == pytest.approx(64 / 16_000 / 60)
    assert stats["test"]["annotation_minutes"] == pytest.approx(64 / 16_000 / 60)


def test_npy_dir_include_labels_remaps_targets_and_noise(tmp_path: Path):
    root = tmp_path / "legacy.npy"
    root.mkdir()
    np.save(
        root / "attrs.npy",
        {
            "class_names": ["noise", "pulse", "sine"],
            "class_types": ["segment", "event", "segment"],
            "samplerate_x_Hz": 1_000,
            "samplerate_y_Hz": 1_000,
        },
        allow_pickle=True,
    )
    split_dir = root / "train"
    split_dir.mkdir()
    x = np.zeros((64,), dtype=np.float32)
    y = np.zeros((64, 3), dtype=np.float32)
    y[:, 0] = 1.0
    y[10:20, 0] = 0.0
    y[10:20, 1] = 1.0
    y[30:40, 0] = 0.0
    y[30:40, 2] = 1.0
    np.save(split_dir / "x.npy", x)
    np.save(split_dir / "y.npy", y)

    datamodule = NPYDirDataModule(
        data_dir=str(root),
        batch_size=1,
        num_time_steps=64,
        chunk_stride=64,
        hop_s=1 / 1_000,
        include_labels=["pulse"],
        num_workers=0,
        persistent_workers=False,
    )
    _, _, targets, _ = next(iter(datamodule.train_dataloader()))

    assert datamodule.class_names == ["noise", "pulse"]
    assert datamodule.class_types == ["segment", "event"]
    assert targets.shape == (1, 64, 2)
    assert torch.all(targets[0, 10:20, 1] == 1.0)
    assert torch.all(targets[0, 30:40, 1] == 0.0)
    assert torch.all(targets[0, 30:40, 0] == 1.0)


def test_npy_dir_include_labels_collapses_selected_targets_when_ignoring_names(tmp_path: Path):
    root = tmp_path / "legacy.npy"
    root.mkdir()
    np.save(
        root / "attrs.npy",
        {
            "class_names": ["noise", "pulse", "sine"],
            "samplerate_x_Hz": 1_000,
            "samplerate_y_Hz": 1_000,
        },
        allow_pickle=True,
    )
    split_dir = root / "train"
    split_dir.mkdir()
    x = np.zeros((64,), dtype=np.float32)
    y = np.zeros((64, 3), dtype=np.float32)
    y[:, 0] = 1.0
    y[10:20, 0] = 0.0
    y[10:20, 1] = 1.0
    y[30:40, 0] = 0.0
    y[30:40, 2] = 1.0
    np.save(split_dir / "x.npy", x)
    np.save(split_dir / "y.npy", y)

    datamodule = NPYDirDataModule(
        data_dir=str(root),
        batch_size=1,
        num_time_steps=64,
        chunk_stride=64,
        hop_s=1 / 1_000,
        ignore_class_names=True,
        include_labels=["pulse"],
        num_workers=0,
        persistent_workers=False,
    )
    _, _, targets, _ = next(iter(datamodule.train_dataloader()))

    assert datamodule.class_names == ["noise", "song"]
    assert torch.all(targets[0, 10:20, 1] == 1.0)
    assert torch.all(targets[0, 30:40, 1] == 0.0)
    assert torch.all(targets[0, 30:40, 0] == 1.0)
