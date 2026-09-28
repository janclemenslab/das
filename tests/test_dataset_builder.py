import numpy as np
import soundfile as sf

from das.data.dataset_builder import make_training_dataset
from das.data.npy_dir import NPYDirDataModule, is_npy_dir


def test_make_training_dataset_from_annotated_wav(tmp_path):
    source = tmp_path / "recordings"
    source.mkdir()
    sf.write(source / "clip.wav", np.zeros(1_000, dtype=np.float32), 1_000)
    (source / "clip_annotations.csv").write_text(
        "name,start_seconds,stop_seconds\npulse,0.1,0.1\nsine,0.3,0.5\n",
        encoding="utf-8",
    )

    output = make_training_dataset(source, tmp_path / "training.npy", seed=1)

    assert is_npy_dir(output)
    data = NPYDirDataModule(data_dir=str(output), batch_size=1)
    assert data.class_names == ["noise", "pulse", "sine"]
    assert data.class_types == ["segment", "event", "segment"]
    assert sum(np.load(output / split / "x.npy").shape[0] for split in ("train", "val", "test")) == 1_000
    assert np.load(output / "train" / "y.npy")[:, 1].sum() > 0
    assert np.load(output / "train" / "y.npy")[:, 2].sum() > 0


def test_make_training_dataset_splits_multichannel_wav_and_npz_files(tmp_path):
    source = tmp_path / "recordings"
    source.mkdir()
    for index in range(3):
        stem = f"clip{index}"
        audio = np.zeros((1_000, 2), dtype=np.float32)
        if index == 0:
            np.savez(source / f"{stem}.npz", data=audio, samplerate=1_000)
        else:
            sf.write(source / f"{stem}.wav", audio, 1_000)
        (source / f"{stem}_annotations.csv").write_text(
            "name,start_seconds,stop_seconds\nsine,0.2,0.4\n", encoding="utf-8"
        )

    output = make_training_dataset(
        source, tmp_path / "training.npy", split_by="files",
        validation_fraction=1 / 3, test_fraction=1 / 3, seed=1,
    )

    for split in ("train", "val", "test"):
        assert np.load(output / split / "x.npy").shape == (1_000, 2)
        assert np.load(output / split / "y.npy").shape == (1_000, 2)
