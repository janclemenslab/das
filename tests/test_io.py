import numpy as np

from das import io


def test_load_npy_dir(tmp_path):
    dataset_path = tmp_path / "dataset.npy"

    data = io.npy_dir.NpyDir()
    data["train"] = {
        "x": np.arange(24, dtype=np.float32).reshape(12, 2),
        "y": np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), (6, 1)),
    }
    data["val"] = {
        "x": np.arange(12, dtype=np.float32).reshape(6, 2),
        "y": np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), (3, 1)),
    }
    data["test"] = {
        "x": np.arange(12, dtype=np.float32).reshape(6, 2),
        "y": np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), (3, 1)),
    }
    data.attrs = {
        "samplerate_x_Hz": 10_000,
        "samplerate_y_Hz": 10_000,
        "class_names": ["noise", "pulse"],
        "class_types": ["segment", "event"],
    }

    data.save(dataset_path)
    loaded_direct = io.npy_dir.NpyDir.load(str(dataset_path))
    loaded = io.load(str(dataset_path))

    assert loaded_direct["train"]["x"].shape == (12, 2)
    assert loaded_direct.attrs["class_names"] == ["noise", "pulse"]
    assert loaded["train"]["x"].shape == (12, 2)
    assert loaded["train"]["y"].shape == (12, 2)
    assert loaded.attrs["class_names"] == ["noise", "pulse"]


def test_audio_sequence_shapes():
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), (6, 1))

    seq = io.AudioSequence(x=x, y=y, batch_size=2, nb_hist=4, shuffle=False)
    batch_x, batch_y = seq[0]

    assert batch_x.shape == (2, 4, 2)
    assert batch_y.shape == (2, 2)
    assert batch_x.dtype == np.float32
    assert batch_y.dtype == np.float32


def test_legacy_npy_dir_api(tmp_path):
    from das import npy_dir

    data = npy_dir.DictClass({"train": {"x": np.arange(6)}})
    data.attrs = {"samplerate_x_Hz": 10_000}
    path = tmp_path / "legacy.npy"

    npy_dir.save(path, data)
    loaded = npy_dir.load(path)

    np.testing.assert_array_equal(loaded["train"]["x"], data["train"]["x"])
    assert loaded.attrs == data.attrs
