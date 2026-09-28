from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import zarr

import das.data.audio_dir as audio_dir
from das.data import AudioDirDataModule, resolve_training_data_dir
from das.data.audio_dir import AudioDirGenerator, audio_file_info, load_audio_array


ZF_WAVS_DIR = Path(__file__).parent / "fixtures" / "zf" / "audio_dir" / "zf_wavs"


def _write_audio(path: Path, *, samplerate: int = 16_000, samples: int = 2_048) -> Path:
    sf.write(path, np.zeros(samples, dtype=np.float32), samplerate)
    return path


def _write_h5_audio(path: Path, data: np.ndarray, *, samplerate: int = 16_000, rate_key: str = "rate") -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples", data=data)
        handle.attrs[rate_key] = samplerate
    return path


def _write_annotations(path: Path, *, name: str = "song") -> Path:
    annotations = pd.DataFrame(
        [
            {
                "name": name,
                "start_seconds": 0.08,
                "stop_seconds": 0.09,
            }
        ]
    )
    annotations.to_csv(path, index=False)
    return path


def _build_datamodule(tmp_path: Path, *, chunk_stride: int | None = None) -> AudioDirDataModule:
    return AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=chunk_stride,
        num_workers=0,
        persistent_workers=False,
    )


def test_file_split_seed_is_independent_of_global_rng(tmp_path: Path):
    for index in range(8):
        audio_path = _write_audio(tmp_path / f"annotated-{index}.wav")
        _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    first = AudioDirDataModule(
        data_dir=str(tmp_path), batch_size=1, num_time_steps=512, val_ratio=0.25, test_ratio=0.25, split_seed=7
    )
    torch.rand(10_000)
    second = AudioDirDataModule(
        data_dir=str(tmp_path), batch_size=1, num_time_steps=512, val_ratio=0.25, test_ratio=0.25, split_seed=7
    )

    assert {name: subset["audio"] for name, subset in first.subsets.items()} == {
        name: subset["audio"] for name, subset in second.subsets.items()
    }


def test_audio_dir_training_batches_include_targets(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    datamodule = _build_datamodule(tmp_path)
    batch = next(iter(datamodule.train_dataloader()))

    assert len(batch) == 4
    inputs, input_lengths, targets, target_lengths = batch

    assert inputs.shape == (1, 512)
    assert torch.equal(input_lengths, torch.tensor([512]))
    assert targets.shape[0] == 1
    assert targets.shape[-1] == datamodule.num_classes
    assert torch.equal(target_lengths, torch.tensor([33]))


def test_clipping_to_overlapping_chunk_intervals_does_not_duplicate_annotations():
    annotations = pd.DataFrame(
        [{"name": "pulse", "start_seconds": 0.75, "stop_seconds": 0.75}]
    )

    clipped = audio_dir._annotations_clipped_to_intervals(
        annotations,
        [(0.0, 1.0), (0.5, 1.5)],
    )

    assert clipped.to_dict("records") == annotations.to_dict("records")


def test_audio_dir_predict_batches_do_not_require_annotations(tmp_path: Path):
    _write_audio(tmp_path / "unlabeled.wav")

    datamodule = _build_datamodule(tmp_path)
    predict_loader = datamodule.predict_dataloader()
    batch = next(iter(predict_loader))

    assert len(batch) == 2
    inputs, input_lengths = batch

    assert len(predict_loader.dataset) > 0
    assert inputs.shape == (1, 512)
    assert torch.equal(input_lengths, torch.tensor([512]))


def test_audio_dir_predict_resamples_to_target_samplerate(tmp_path: Path):
    sf.write(tmp_path / "unlabeled.wav", np.linspace(-1.0, 1.0, 16, dtype=np.float32), 1_000)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=8,
        chunk_stride=8,
        target_samplerate=500,
        num_workers=0,
        persistent_workers=False,
    )
    predict_loader = datamodule.predict_dataloader()
    inputs, input_lengths = next(iter(predict_loader))

    assert datamodule.source_samplerates == {1_000}
    assert predict_loader.dataset.nb_samples_in_file.tolist() == [8]
    assert predict_loader.dataset.samplerate_per_file.tolist() == [500]
    assert inputs.shape == (1, 8)
    assert torch.equal(input_lengths, torch.tensor([8]))


def test_audio_dir_predict_batches_use_all_audio_without_annotations(tmp_path: Path):
    annotated_audio = _write_audio(tmp_path / "annotated.wav")
    _write_annotations(tmp_path / f"{annotated_audio.stem}_annotations.csv")
    unlabeled_audio = _write_audio(tmp_path / "unlabeled.wav")

    datamodule = _build_datamodule(tmp_path)
    predict_loader = datamodule.predict_dataloader()
    batch = next(iter(predict_loader))

    assert len(batch) == 2
    inputs, input_lengths = batch
    assert set(predict_loader.dataset.audio_files) == {annotated_audio, unlabeled_audio}
    assert len(predict_loader.dataset) > len(datamodule.train_dataloader().dataset)
    assert inputs.shape == (1, 512)
    assert torch.equal(input_lengths, torch.tensor([512]))


def test_audio_dir_generator_opens_one_audio_file_at_a_time(monkeypatch: pytest.MonkeyPatch):
    state = {"open": 0, "max_open": 0}

    class FakeAudio:
        samplerate = 16_000
        channels = 1

        def __init__(self, path: Path):
            self.path = Path(path)
            self.closed = False
            state["open"] += 1
            state["max_open"] = max(state["max_open"], state["open"])

        def __len__(self):
            return 1_024

        def read(self, start: int, frames: int):
            return np.zeros(frames, dtype=np.float32)

        def close(self):
            if not self.closed:
                self.closed = True
                state["open"] -= 1

    monkeypatch.setattr(audio_dir, "open_audio_file", FakeAudio)

    dataset = AudioDirGenerator(
        num_time_steps=512,
        chunk_stride=512,
        audio_files=[Path(f"audio_{idx}.wav") for idx in range(10)],
        return_targets=False,
    )

    assert state == {"open": 0, "max_open": 1}

    state["max_open"] = 0
    inputs, input_length = dataset[0]

    assert inputs.shape == (512,)
    assert input_length == 512
    assert state == {"open": 0, "max_open": 1}


def test_audio_dir_predicts_h5_audio_from_samples_dataset(tmp_path: Path):
    data = np.stack([np.arange(1024), np.arange(1024) + 10_000], axis=1).astype(np.float32)
    audio_path = _write_h5_audio(tmp_path / "recording.h5", data, samplerate=10_000)

    datamodule = _build_datamodule(tmp_path)
    dataset = datamodule.predict_dataloader().dataset
    inputs, input_lengths = dataset[0]

    assert datamodule.all_audio_files == [audio_path]
    assert dataset.samplerate_per_file.tolist() == [10_000]
    assert dataset.nb_samples_in_file.tolist() == [1024]
    assert inputs.shape == (2, 512)
    assert input_lengths == 512
    np.testing.assert_array_equal(inputs[0, :3], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(inputs[1, :3], np.array([10_000, 10_001, 10_002], dtype=np.float32))


def test_audio_dir_accepts_transposed_h5_audio_and_samplerate_dataset(tmp_path: Path):
    data = np.stack([np.arange(1024), np.arange(1024) + 20_000], axis=0).astype(np.float32)
    with h5py.File(tmp_path / "recording.h5", "w") as handle:
        handle.create_dataset("audio", data=data)
        handle.create_dataset("sample_rate_Hz", data=np.array(12_345))

    datamodule = _build_datamodule(tmp_path)
    dataset = datamodule.predict_dataloader().dataset
    inputs, input_lengths = dataset[0]

    assert dataset.samplerate_per_file.tolist() == [12_345]
    assert dataset.nb_samples_in_file.tolist() == [1024]
    assert inputs.shape == (2, 512)
    assert input_lengths == 512
    np.testing.assert_array_equal(inputs[0, :3], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(inputs[1, :3], np.array([20_000, 20_001, 20_002], dtype=np.float32))


def test_audio_dir_accepts_non_wav_soundfile_audio(tmp_path: Path):
    audio_path = tmp_path / "unlabeled.flac"
    sf.write(audio_path, np.zeros(1024, dtype=np.float32), 8_000)

    datamodule = _build_datamodule(tmp_path)

    assert datamodule.all_audio_files == [audio_path]
    assert datamodule.predict_dataloader().dataset.samplerate_per_file.tolist() == [8_000]


def test_audio_dir_accepts_npz_audio_with_samplerate(tmp_path: Path):
    audio_path = tmp_path / "recording.npz"
    data = np.stack([np.arange(1024), np.arange(1024) + 100], axis=1).astype(np.float32)
    np.savez(audio_path, data=data, samplerate=np.array([22_050]))

    datamodule = _build_datamodule(tmp_path)
    inputs, input_lengths = datamodule.predict_dataloader().dataset[0]

    assert datamodule.all_audio_files == [audio_path]
    assert input_lengths == 512
    assert inputs.shape == (2, 512)
    np.testing.assert_array_equal(inputs[0, :3], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(inputs[1, :3], np.array([100, 101, 102], dtype=np.float32))


def test_audio_dir_accepts_npy_audio_with_sidecar_samplerate(tmp_path: Path):
    audio_path = tmp_path / "recording.npy"
    np.save(audio_path, np.arange(1024, dtype=np.float32))
    (tmp_path / "recording.npy.audio.json").write_text('{"samplerate_hz": 12345}', encoding="utf-8")

    info = audio_file_info(audio_path)
    audio, samplerate = load_audio_array(audio_path)

    assert info == {"frames": 1024, "samplerate": 12345, "channels": 1}
    assert samplerate == 12345
    np.testing.assert_array_equal(audio[:3], np.array([0, 1, 2], dtype=np.float32))


def test_audio_dir_accepts_mmap_audio_with_sidecar_metadata(tmp_path: Path):
    audio_path = tmp_path / "recording.mmap"
    data = np.arange(1024 * 2, dtype=np.int16).reshape(1024, 2)
    mmap = np.memmap(audio_path, mode="w+", dtype="int16", shape=data.shape)
    mmap[:] = data
    mmap.flush()
    del mmap
    (tmp_path / "recording.mmap.audio.json").write_text(
        '{"samplerate_hz": 250000, "dtype": "int16", "shape": [1024, 2]}',
        encoding="utf-8",
    )

    datamodule = _build_datamodule(tmp_path)
    inputs, input_lengths = datamodule.predict_dataloader().dataset[0]

    assert datamodule.all_audio_files == [audio_path]
    assert datamodule.input_num_channels == 2
    assert input_lengths == 512
    assert inputs.shape == (2, 512)
    np.testing.assert_array_equal(inputs[0, :3], data[:3, 0].astype(np.float32))
    np.testing.assert_array_equal(inputs[1, :3], data[:3, 1].astype(np.float32))


def test_audio_dir_allows_mixed_channel_counts_with_batch_size_one(tmp_path: Path):
    sf.write(tmp_path / "nine.wav", np.zeros((1024, 9), dtype=np.float32), 10_000)
    sf.write(tmp_path / "sixteen.wav", np.zeros((1024, 16), dtype=np.float32), 10_000)

    datamodule = _build_datamodule(tmp_path)
    batches = list(datamodule.predict_dataloader())

    assert datamodule.input_num_channels == 16
    assert [batch[0].shape for batch in batches] == [(1, 9, 512)] * 3 + [(1, 16, 512)] * 3

    with pytest.raises(ValueError, match="Mixed channel counts require batch_size=1"):
        AudioDirDataModule(
            data_dir=str(tmp_path),
            batch_size=2,
            num_time_steps=512,
            num_workers=0,
            persistent_workers=False,
        )


def test_audio_dir_accepts_zarr_audio_with_sidecar_samplerate_and_dataset(tmp_path: Path):
    store_path = tmp_path / "recording.zarr"
    data = np.stack([np.arange(1024), np.arange(1024) + 500], axis=1).astype(np.float32)
    root = zarr.open_group(store_path, mode="w")
    root.create_dataset("nested/audio", data=data, shape=data.shape, dtype=data.dtype)
    (tmp_path / "recording.zarr.audio.json").write_text(
        '{"samplerate_hz": 32000, "audio_dataset": "nested/audio"}',
        encoding="utf-8",
    )

    datamodule = _build_datamodule(tmp_path)
    inputs, input_lengths = datamodule.predict_dataloader().dataset[0]

    assert datamodule.all_audio_files == [store_path]
    assert input_lengths == 512
    assert inputs.shape == (2, 512)
    np.testing.assert_array_equal(inputs[0, :3], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(inputs[1, :3], np.array([500, 501, 502], dtype=np.float32))


def test_audio_dir_default_chunk_stride_uses_half_window_overlap(tmp_path: Path):
    _write_audio(tmp_path / "unlabeled.wav", samples=2_048)

    datamodule = _build_datamodule(tmp_path, chunk_stride=None)
    dataset = datamodule.predict_dataloader().dataset

    assert dataset.chunk_stride == 256
    assert len(dataset) == 7


def test_audio_dir_overlapping_chunk_stride_increases_chunk_count(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=2_048)
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    datamodule = _build_datamodule(tmp_path, chunk_stride=256)

    assert datamodule.predict_dataloader().dataset.chunk_stride == 256
    assert len(datamodule.predict_dataloader().dataset) == 7
    assert len(datamodule.train_dataloader().dataset) == 7


def test_audio_dir_without_annotation_markers_uses_whole_file(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=2_048)
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset

    assert dataset.chunk_start_samples.tolist() == [0, 512, 1024, 1536]
    assert dataset.chunk_input_lengths.tolist() == [512, 512, 512, 512]


def test_audio_dir_annotation_markers_limit_supervised_chunks(tmp_path: Path):
    samplerate = 1_000
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=2_000, samplerate=samplerate)
    pd.DataFrame(
        [
            {"name": "annotation_start", "start_seconds": 0.5, "stop_seconds": 0.5},
            {"name": "song", "start_seconds": 0.6, "stop_seconds": 0.7},
            {"name": "annotation_end", "start_seconds": 1.3, "stop_seconds": 1.3},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=400,
        chunk_stride=400,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset

    assert datamodule.class_names == ["noise", "song"]
    assert dataset.chunk_start_samples.tolist() == [500, 900]
    assert dataset.chunk_input_lengths.tolist() == [400, 400]
    assert len(datamodule.predict_dataloader().dataset) > len(dataset)


def test_audio_dir_annotation_marker_duplicate_rules(tmp_path: Path):
    samplerate = 1_000
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=2_000, samplerate=samplerate)
    pd.DataFrame(
        [
            {"name": "annotation_end", "start_seconds": 0.2, "stop_seconds": 0.2},
            {"name": "annotation_end", "start_seconds": 0.3, "stop_seconds": 0.3},
            {"name": "annotation_start", "start_seconds": 0.5, "stop_seconds": 0.5},
            {"name": "annotation_start", "start_seconds": 0.6, "stop_seconds": 0.6},
            {"name": "song", "start_seconds": 0.7, "stop_seconds": 0.8},
            {"name": "annotation_end", "start_seconds": 0.9, "stop_seconds": 0.9},
            {"name": "annotation_end", "start_seconds": 1.0, "stop_seconds": 1.0},
            {"name": "annotation_start", "start_seconds": 1.5, "stop_seconds": 1.5},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=300,
        chunk_stride=300,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset

    assert dataset.chunk_start_samples.tolist() == [0, 500, 700, 1500, 1700]
    assert dataset.chunk_input_lengths.tolist() == [300, 300, 300, 300, 300]


def test_audio_dir_short_annotation_interval_is_padded_without_outside_audio(tmp_path: Path):
    samplerate = 1_000
    audio_path = tmp_path / "annotated.wav"
    audio = np.arange(1_000, dtype=np.float32)
    sf.write(audio_path, audio, samplerate, subtype="FLOAT")
    pd.DataFrame(
        [
            {"name": "annotation_start", "start_seconds": 0.2, "stop_seconds": 0.2},
            {"name": "song", "start_seconds": 0.21, "stop_seconds": 0.22},
            {"name": "annotation_end", "start_seconds": 0.25, "stop_seconds": 0.25},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=100,
        chunk_stride=100,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset
    inputs, input_length, targets, target_length = dataset[0]

    assert dataset.chunk_start_samples.tolist() == [200]
    assert input_length == 50
    assert target_length == 50
    np.testing.assert_array_equal(inputs[:50], audio[200:250])
    np.testing.assert_array_equal(inputs[50:], np.zeros(50, dtype=np.float32))
    assert np.any(targets[:, 1] == 1.0)


def test_audio_dir_marker_rows_do_not_count_as_annotations(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=1_000, samplerate=1_000)
    pd.DataFrame(
        [
            {"name": "annotation_start", "start_seconds": 0.1, "stop_seconds": 0.1},
            {"name": "song", "start_seconds": 0.2, "stop_seconds": 0.3},
            {"name": "annotation_end", "start_seconds": 0.8, "stop_seconds": 0.8},
            {"name": "outside", "start_seconds": 0.9, "stop_seconds": 0.95},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=100,
        chunk_stride=100,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    stats = {row["split"]: row for row in datamodule.split_stats()}

    assert datamodule.class_names == ["noise", "song"]
    assert stats["train"]["annotation_count"] == 1
    assert stats["train"]["annotation_minutes"] == pytest.approx(0.1 / 60)
    assert stats["train"]["audio_minutes"] == pytest.approx(0.7 / 60)


def test_audio_dir_shared_filepath_markers_are_per_audio(tmp_path: Path):
    samplerate = 1_000
    first_audio = _write_audio(tmp_path / "first.wav", samples=1_000, samplerate=samplerate)
    second_audio = _write_audio(tmp_path / "second.wav", samples=1_000, samplerate=samplerate)
    pd.DataFrame(
        [
            {"filepath": first_audio.name, "name": "annotation_start", "start_seconds": 0.1, "stop_seconds": 0.1},
            {"filepath": first_audio.name, "name": "alpha", "start_seconds": 0.2, "stop_seconds": 0.3},
            {"filepath": first_audio.name, "name": "annotation_end", "start_seconds": 0.4, "stop_seconds": 0.4},
            {"filepath": second_audio.name, "name": "annotation_start", "start_seconds": 0.6, "stop_seconds": 0.6},
            {"filepath": second_audio.name, "name": "zebra", "start_seconds": 0.65, "stop_seconds": 0.7},
            {"filepath": second_audio.name, "name": "annotation_end", "start_seconds": 0.8, "stop_seconds": 0.8},
        ]
    ).to_csv(tmp_path / "annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=100,
        chunk_stride=100,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    starts_by_audio = {
        path.name: starts.tolist()
        for path, starts in zip(datamodule.annotated_audio_files, datamodule.annotation_chunk_starts, strict=True)
    }

    assert datamodule.class_names == ["noise", "alpha", "zebra"]
    assert starts_by_audio == {"first.wav": [100, 200, 300], "second.wav": [600, 700]}


def test_audio_dir_can_split_single_file_within_chunks(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=5_120)
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        split_within_files=True,
        num_workers=0,
        persistent_workers=False,
    )

    assert {name: len(subset["audio"]) for name, subset in datamodule.subsets.items()} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert len(datamodule.train_dataloader().dataset) == 6
    assert len(datamodule.val_dataloader().dataset) == 2
    assert len(datamodule.test_dataloader().dataset) == 2
    assert datamodule.train_dataloader().dataset.chunk_start_samples.tolist() == [0, 512, 1024, 1536, 2048, 2560]
    assert datamodule.val_dataloader().dataset.chunk_start_samples.tolist() == [3072, 3584]
    assert datamodule.test_dataloader().dataset.chunk_start_samples.tolist() == [4096, 4608]


def test_audio_dir_within_file_split_drops_overlapping_boundary_chunks(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=5_120)
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=None,
        split_within_files=True,
        num_workers=0,
        persistent_workers=False,
    )

    train_starts = datamodule.train_dataloader().dataset.chunk_start_samples
    val_starts = datamodule.val_dataloader().dataset.chunk_start_samples
    test_starts = datamodule.test_dataloader().dataset.chunk_start_samples

    assert train_starts[-1] + datamodule.num_time_steps <= val_starts[0]
    assert val_starts[-1] + datamodule.num_time_steps <= test_starts[0]


def test_audio_dir_split_stats_report_files_and_minutes(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=16_000, samplerate=16_000)
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv")

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    stats = {row["split"]: row for row in datamodule.split_stats()}

    assert stats["train"]["audio_file_count"] == 1
    assert stats["train"]["audio_minutes"] == pytest.approx(1 / 60)
    assert stats["train"]["annotation_file_count"] == 1
    assert stats["train"]["annotation_count"] == 1
    assert stats["train"]["annotation_minutes"] == pytest.approx(0.01 / 60)
    assert stats["val"]["audio_file_count"] == 0
    assert stats["test"]["audio_file_count"] == 0


def test_audio_dir_within_file_split_stats_report_split_minutes_and_annotation_counts(tmp_path: Path):
    samplerate = 1_000
    audio_path = _write_audio(tmp_path / "annotated.wav", samples=10 * samplerate, samplerate=samplerate)
    pd.DataFrame(
        [
            {"name": "song", "start_seconds": float(second), "stop_seconds": float(second) + 0.1}
            for second in range(10)
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=samplerate,
        chunk_stride=samplerate,
        split_within_files=True,
        num_workers=0,
        persistent_workers=False,
    )
    stats = {row["split"]: row for row in datamodule.split_stats()}

    assert stats["train"]["audio_file_count"] == 1
    assert stats["val"]["audio_file_count"] == 1
    assert stats["test"]["audio_file_count"] == 1
    assert stats["train"]["audio_minutes"] == pytest.approx(6 / 60)
    assert stats["val"]["audio_minutes"] == pytest.approx(2 / 60)
    assert stats["test"]["audio_minutes"] == pytest.approx(2 / 60)
    assert stats["train"]["annotation_count"] == 6
    assert stats["val"]["annotation_count"] == 2
    assert stats["test"]["annotation_count"] == 2
    assert stats["train"]["annotation_minutes"] == pytest.approx(0.6 / 60)
    assert stats["val"]["annotation_minutes"] == pytest.approx(0.2 / 60)
    assert stats["test"]["annotation_minutes"] == pytest.approx(0.2 / 60)


def test_audio_dir_includes_tail_chunk_when_stride_does_not_land_on_file_end(tmp_path: Path):
    _write_audio(tmp_path / "unlabeled.wav", samples=1_000)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=300,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.predict_dataloader().dataset

    assert dataset.chunk_start_samples.tolist() == [0, 300, 488]
    assert dataset.chunk_input_lengths.tolist() == [512, 512, 512]


def test_audio_dir_supervised_stages_require_annotations(tmp_path: Path):
    _write_audio(tmp_path / "unlabeled.wav")

    datamodule = _build_datamodule(tmp_path)

    with pytest.raises(ValueError, match="No annotated audio files were found"):
        datamodule.train_dataloader()


def test_audio_dir_mixed_directories_use_all_audio_for_predict_only(tmp_path: Path):
    annotated_audio = _write_audio(tmp_path / "annotated.wav")
    _write_annotations(tmp_path / f"{annotated_audio.stem}_annotations.csv")
    unlabeled_audio = _write_audio(tmp_path / "unlabeled.wav")

    datamodule = _build_datamodule(tmp_path)
    train_loader = datamodule.train_dataloader()
    predict_loader = datamodule.predict_dataloader()

    assert set(datamodule.all_audio_files) == {annotated_audio, unlabeled_audio}
    assert set(datamodule.annotated_audio_files) == {annotated_audio}
    assert set(train_loader.dataset.audio_files).issubset(set(datamodule.annotated_audio_files))
    assert set(predict_loader.dataset.audio_files) == set(datamodule.all_audio_files)
    assert len(predict_loader.dataset) > len(train_loader.dataset)


def test_audio_dir_accepts_shared_annotations_with_filepath_column(tmp_path: Path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    first_audio = _write_audio(nested_dir / "first.wav")
    second_audio = _write_audio(tmp_path / "second.wav")
    annotation_file = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {"filepath": "nested/first.wav", "name": "alpha", "start_seconds": 0.01, "stop_seconds": 0.02},
            {"filepath": second_audio.name, "name": "zebra", "start_seconds": 0.03, "stop_seconds": 0.04},
        ]
    ).to_csv(annotation_file, index=False)

    datamodule = _build_datamodule(tmp_path)

    assert {path.resolve() for path in datamodule.all_audio_files} == {first_audio.resolve(), second_audio.resolve()}
    assert {path.resolve() for path in datamodule.annotated_audio_files} == {first_audio.resolve(), second_audio.resolve()}
    assert set(datamodule.annotation_files) == {annotation_file}
    assert datamodule.class_names == ["noise", "alpha", "zebra"]
    assert sorted(annotation["name"].iloc[0] for annotation in datamodule.annotations) == ["alpha", "zebra"]


def test_audio_dir_finds_recursive_json_labels(tmp_path: Path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    audio_path = _write_audio(nested_dir / "call.wav")
    (nested_dir / "call.json").write_text(
        '{"onset": [0.01], "offset": [0.02], "cluster": ["call"]}',
        encoding="utf-8",
    )

    datamodule = _build_datamodule(tmp_path)

    assert datamodule.annotated_audio_files == [audio_path]
    assert datamodule.class_names == ["noise", "call"]
    assert datamodule.annotations[0].to_dict("records") == [
        {
            "onset": 0.01,
            "offset": 0.02,
            "cluster": "call",
            "name": "call",
            "start_seconds": 0.01,
            "stop_seconds": 0.02,
        }
    ]


def test_audio_dir_accepts_whisperseg_style_csv_labels(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "call.wav")
    pd.DataFrame({"onset": [0.01], "offset": [0.02], "cluster": ["call"]}).to_csv(
        tmp_path / "call.csv",
        index=False,
    )

    datamodule = _build_datamodule(tmp_path)

    assert datamodule.annotated_audio_files == [audio_path]
    assert datamodule.class_names == ["noise", "call"]


def test_resolve_training_data_dir_downloads_custom_hf_dataset(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return "/tmp/hf-cache/vad-zebra-finch"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))

    resolved = resolve_training_data_dir("someone/custom-dataset")

    assert resolved == "/tmp/hf-cache/vad-zebra-finch"
    assert calls["repo_id"] == "someone/custom-dataset"
    assert calls["repo_type"] == "dataset"
    assert "*.json" in calls["allow_patterns"]
    assert "*.wav" in calls["allow_patterns"]


def test_audio_dir_infers_class_names_in_sorted_order(tmp_path: Path):
    first_audio = _write_audio(tmp_path / "first.wav")
    pd.DataFrame(
        [{"name": "zebra", "start_seconds": 0.01, "stop_seconds": 0.02}]
    ).to_csv(tmp_path / f"{first_audio.stem}_annotations.csv", index=False)

    second_audio = _write_audio(tmp_path / "second.wav")
    pd.DataFrame(
        [{"name": "alpha", "start_seconds": 0.03, "stop_seconds": 0.04}]
    ).to_csv(tmp_path / f"{second_audio.stem}_annotations.csv", index=False)

    datamodule = _build_datamodule(tmp_path)

    assert datamodule.class_names == ["noise", "alpha", "zebra"]


def test_audio_dir_include_labels_treats_deselected_labels_as_noise(tmp_path: Path):
    samplerate = 1_000
    audio_path = _write_audio(tmp_path / "annotated.wav", samplerate=samplerate, samples=1_000)
    pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.1, "stop_seconds": 0.2},
            {"name": "sine", "start_seconds": 0.3, "stop_seconds": 0.4},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=100,
        chunk_stride=100,
        hop_s=1 / samplerate,
        include_labels=["pulse"],
        val_ratio=0,
        test_ratio=0,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset

    _, _, pulse_targets, _ = dataset[1]
    _, _, sine_targets, _ = dataset[3]

    assert datamodule.class_names == ["noise", "pulse"]
    assert pulse_targets[:, 1].sum() > 0
    assert sine_targets[:, 1].sum() == 0
    assert np.all(sine_targets[:, 0] == 1.0)


def test_audio_dir_include_labels_rejects_unknown_labels(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    _write_annotations(tmp_path / f"{audio_path.stem}_annotations.csv", name="pulse")

    with pytest.raises(ValueError, match="missing"):
        AudioDirDataModule(
            data_dir=str(tmp_path),
            include_labels=["missing"],
            num_workers=0,
            persistent_workers=False,
        )


def test_audio_dir_labels_are_offset_from_chunk_start(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    pd.DataFrame(
        [{"name": "song", "start_seconds": 0.040, "stop_seconds": 0.050}]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        hop_s=1 / 16_000,
        class_names=["noise", "song"],
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset

    _, _, targets, _ = dataset[1]
    song_indices = np.flatnonzero(targets[:, 1])

    assert song_indices[0] == 128
    assert song_indices[-1] in {287, 288}


def test_audio_dir_zero_duration_annotations_get_minimum_target_box(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    pd.DataFrame(
        [{"name": "pulse", "start_seconds": 0.040, "stop_seconds": 0.040}]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        hop_s=1 / 16_000,
        class_names=["noise", "pulse"],
        min_annotation_duration_s=0.004,
        num_workers=0,
        persistent_workers=False,
    )
    dataset = datamodule.train_dataloader().dataset

    _, _, targets, _ = dataset[1]
    pulse_indices = np.flatnonzero(targets[:, 1])

    assert datamodule.class_types == ["segment", "event"]
    assert pulse_indices[0] == 96
    assert pulse_indices[-1] == 159
    assert len(pulse_indices) == 64


def test_audio_dir_infers_events_only_for_all_zero_duration_labels(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.040, "stop_seconds": 0.040},
            {"name": "sine", "start_seconds": 0.080, "stop_seconds": 0.120},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        hop_s=1 / 16_000,
        num_workers=0,
        persistent_workers=False,
    )

    assert datamodule.class_names == ["noise", "pulse", "sine"]
    assert datamodule.class_types == ["segment", "event", "segment"]


def test_audio_dir_include_labels_preserves_event_class_type(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.040, "stop_seconds": 0.040},
            {"name": "sine", "start_seconds": 0.080, "stop_seconds": 0.120},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        hop_s=1 / 16_000,
        include_labels=["pulse"],
        num_workers=0,
        persistent_workers=False,
    )

    assert datamodule.class_names == ["noise", "pulse"]
    assert datamodule.class_types == ["segment", "event"]


def test_audio_dir_mixed_duration_label_is_segment(tmp_path: Path):
    audio_path = _write_audio(tmp_path / "annotated.wav")
    pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.040, "stop_seconds": 0.040},
            {"name": "pulse", "start_seconds": 0.080, "stop_seconds": 0.120},
        ]
    ).to_csv(tmp_path / f"{audio_path.stem}_annotations.csv", index=False)

    datamodule = AudioDirDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_time_steps=512,
        chunk_stride=512,
        hop_s=1 / 16_000,
        num_workers=0,
        persistent_workers=False,
    )

    assert datamodule.class_names == ["noise", "pulse"]
    assert datamodule.class_types == ["segment", "segment"]


def test_audio_dir_zf_fixture_builds_all_splits_and_predict_loader():
    datamodule = AudioDirDataModule(
        data_dir=str(ZF_WAVS_DIR),
        batch_size=2,
        num_time_steps=512,
        num_workers=0,
        persistent_workers=False,
    )

    assert len(datamodule.all_audio_files) == 5
    assert len(datamodule.annotated_audio_files) == 5
    assert len(datamodule.annotation_files) == 5
    assert datamodule.class_names == ["noise", "syll_0", "syll_1", "syll_2", "syll_3", "syll_4", "syll_5"]
    assert {path.parent for path in datamodule.all_audio_files} == {ZF_WAVS_DIR}
    assert {name: len(subset["audio"]) for name, subset in datamodule.subsets.items()} == {"train": 3, "val": 1, "test": 1}

    supervised_audio_files = set()
    supervised_chunk_count = 0
    for stage in ("train", "val", "test"):
        loader = getattr(datamodule, f"{stage}_dataloader")()
        batch = next(iter(loader))
        inputs, input_lengths, targets, target_lengths = batch

        supervised_audio_files.update(loader.dataset.audio_files)
        supervised_chunk_count += len(loader.dataset)

        assert inputs.ndim == 2
        assert inputs.shape[1] == 512
        assert torch.all(input_lengths == 512)
        assert targets.shape == (inputs.shape[0], 17, 7)
        assert torch.all(target_lengths == 17)

    predict_loader = datamodule.predict_dataloader()
    predict_inputs, predict_input_lengths = next(iter(predict_loader))

    assert set(datamodule.all_audio_files) == supervised_audio_files
    assert supervised_chunk_count == len(predict_loader.dataset)
    assert predict_inputs.ndim == 2
    assert predict_inputs.shape[1] == 512
    assert torch.all(predict_input_lengths == 512)
