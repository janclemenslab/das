"""Backward-compatible access to the NPY-directory storage API."""

from .io.npy_dir import NpyDir

DictClass = NpyDir


def load(location, memmap_dirs=None):
    return NpyDir.load(location, memmap_dirs=memmap_dirs)


def save(location, data):
    if not isinstance(data, NpyDir):
        converted = NpyDir(data)
        converted.attrs = getattr(data, "attrs", {})
        data = converted
    data.save(location)
