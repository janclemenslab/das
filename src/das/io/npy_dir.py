"""Dict of dicts <-> hierarchy of npy files."""

import os
import os.path
from glob import glob
from typing import Dict, List, Optional, Union

import numpy as np


class NpyDir(dict):
    """Wrap dict in class so we can attach attrs to it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs: Dict = {}

    def __str__(self):
        out = "Data:\n"
        for top_key in self.keys():
            out = out + f"   {top_key}:\n"
            for key, val in self[top_key].items():
                out = out + f"      {key}: {val.shape}\n"
        out = out + "\nAttributes:\n"
        for key, val in self.attrs.items():
            out = out + f"    {key}: {val}\n"
        return out

    @classmethod
    def load(cls, location: str, memmap_dirs: Optional[Union[List[str], str]] = None) -> "NpyDir":
        """Load hierarchy of npy files into a dict of dicts."""

        if memmap_dirs is None:
            memmap_dirs = ["train"]

        def path_to_key(path):
            return os.path.splitext(os.path.basename(path))[0]

        dir_names = [
            os.path.join(location, name) for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))
        ]
        data = cls()

        attrs_path = os.path.join(location, "attrs.npy")
        if os.path.exists(attrs_path):
            data.attrs = np.load(attrs_path, allow_pickle=True)[()]

        for dir_name in dir_names:
            dir_key = path_to_key(dir_name)
            data[dir_key] = dict()
            npy_files = glob(os.path.join(dir_name, "*.npy"))
            for npy_file in npy_files:
                npy_key = path_to_key(npy_file)
                if memmap_dirs == "all" or dir_key in memmap_dirs:
                    data[dir_key][npy_key] = np.lib.format.open_memmap(npy_file, mode="r")
                else:
                    data[dir_key][npy_key] = np.load(npy_file)
        return data

    def save(self, location: str) -> None:
        """Save a nested dict as a directory hierarchy of npy files."""

        os.makedirs(location, exist_ok=True)
        if hasattr(self, "attrs"):
            np.save(os.path.join(location, "attrs"), dict(self.attrs), allow_pickle=True)

        for key_top in self.keys():
            os.makedirs(os.path.join(location, key_top), exist_ok=True)
            for key, val in self[key_top].items():
                np.save(os.path.join(location, key_top, key + ".npy"), val)
