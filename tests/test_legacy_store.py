from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from das.api import _build_train_datamodule
from das.config import Config
from das.data.legacy_store import is_legacy_store


@pytest.mark.parametrize("suffix", [".h5", ".zarr"])
def test_prebuilt_legacy_dataset_trains_through_npy_reader(tmp_path: Path, suffix: str):
    path = tmp_path / f"data{suffix}"
    opener = h5py.File(path, "w") if suffix == ".h5" else zarr.open(str(path), mode="w")
    if suffix == ".h5":
        context = opener
    else:
        from contextlib import nullcontext
        context = nullcontext(opener)
    with context as store:
        store.attrs["samplerate_x_Hz"] = 1_000
        store.attrs["samplerate_y_Hz"] = 1_000
        store.attrs["classnames"] = ["noise", "song"]
        store.attrs["classtype"] = ["segment", "segment"]
        for split in ("train", "val"):
            group = store.create_group(split)
            group.create_dataset("x", data=np.zeros(128, dtype=np.float32))
            labels = np.zeros((128, 2), dtype=np.float32)
            labels[:, 0] = 1
            group.create_dataset("y", data=labels)

    assert is_legacy_store(path)
    module, samplerate, _ = _build_train_datamodule(Config(mode="train", data_dir=str(path), frontend_type="raw"))
    assert samplerate == 1_000
    assert module.class_names == ["noise", "song"]
    assert len(module.train_dataloader().dataset) > 0
