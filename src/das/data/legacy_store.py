"""Read prebuilt H5/Zarr DAS datasets through the existing npy_dir trainer."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import zarr


def _open_store(path: Path):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return h5py.File(path, "r")
    if path.suffix.lower() == ".zarr":
        return nullcontext(zarr.open(str(path), mode="r"))
    return None


def is_legacy_store(path: str | Path) -> bool:
    context = _open_store(Path(path))
    if context is None:
        return False
    with context as store:
        return "train" in store and "x" in store["train"] and "y" in store["train"]


def materialize_legacy_store(path: str | Path) -> tuple[TemporaryDirectory, Path]:
    source = Path(path)
    directory = TemporaryDirectory(prefix="das-legacy-dataset-")
    destination = Path(directory.name)
    with _open_store(source) as store:
        attrs = dict(store.attrs)
        for old, new in (("classnames", "class_names"), ("classtype", "class_types")):
            if new not in attrs and old in attrs:
                attrs[new] = attrs[old]
        for name in ("class_names", "class_types"):
            if name in attrs:
                attrs[name] = [item.decode() if isinstance(item, bytes) else str(item) for item in attrs[name]]
        np.save(destination / "attrs.npy", attrs, allow_pickle=True)
        for split in ("train", "val", "test"):
            if split not in store:
                continue
            for name in ("x", "y"):
                if name not in store[split]:
                    continue
                source_array = store[split][name]
                output_dir = destination / split
                output_dir.mkdir(exist_ok=True)
                output = np.lib.format.open_memmap(
                    output_dir / f"{name}.npy", mode="w+", dtype=source_array.dtype, shape=source_array.shape
                )
                for start in range(0, source_array.shape[0], 100_000):
                    output[start : start + 100_000] = source_array[start : start + 100_000]
                del output
    return directory, destination
