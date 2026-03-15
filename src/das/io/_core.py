"""I/O and dataset utilities for training and prediction."""

import os.path
from typing import Callable, List, Optional, Sequence

import dask.array
import h5py
import keras
import numpy as np
import zarr
from tqdm.autonotebook import tqdm

from . import npy_dir


class MemoryMappedDirectoryStore(zarr.storage.DirectoryStore):
    # faster access to zarr files via memmaping: https://gist.github.com/ivirshup/5c7df5ed10517abf6567a6a9af6c7eaa
    def _fromfile(self, fn):
        return memoryview(np.memmap(fn, mode="r"))


def _select(data, x_suffix, y_suffix):
    for lvl in ["test", "val", "train"]:
        if lvl in data:
            if "y_" + y_suffix in data[lvl]:
                data[lvl]["y"] = data[lvl]["y_" + y_suffix]
                if "eventtimes_" + y_suffix in data[lvl]:
                    data[lvl]["eventtimes"] = data[lvl]["eventtimes_" + y_suffix]
            if "x_" + x_suffix in data[lvl]:
                data[lvl]["x"] = data[lvl]["x_" + x_suffix]
                if "eventtimes_" + x_suffix in data[lvl]:
                    data[lvl]["eventtimes"] = data[lvl]["eventtimes_" + x_suffix]

    if f"samplerate_x_{x_suffix}_Hz" in data.attrs:
        data.attrs["samplerate_x_Hz"] = data.attrs[f"samplerate_x_{x_suffix}_Hz"]

    if "class_names_" + y_suffix in data.attrs and "class_types_" + y_suffix in data.attrs:
        data.attrs["class_names"] = data.attrs["class_names_" + y_suffix]
        data.attrs["class_types"] = data.attrs["class_types_" + y_suffix]
    return data


def _to_dict(data):
    "Convert dict-like zarr or h5 store `data` to python dictionary."
    d = npy_dir.NpyDir()
    d.attrs = dict(data.attrs)  # cast to dict since data.attrs are read-only for zarr stores
    for key_top in data.keys():
        d[key_top] = dict()
        for key, val in data[key_top].items():
            d[key_top][key] = val
    return d


def load(location, x_suffix="", y_suffix=""):
    """Load data for training/testing from zarr store, npy directory, or hdf5 file."""

    location = os.path.normpath(location)  # remove trailing path separators
    if location.endswith(".zarr"):
        store = zarr.LRUStoreCache(MemoryMappedDirectoryStore(location), max_size=8e9)
        data = zarr.group(store=store, overwrite=False)
    elif location.endswith(".h5"):
        data = h5py.File.open(location, mode="r")
    elif location.endswith(".npy"):
        data = npy_dir.NpyDir.load(location)
    else:
        raise ValueError(
            f'Could not load data. Location {location} has unknown extension - needs to end either in ".zarr", ".npy", or ".h5".'
        )

    data = _to_dict(data)

    if len(x_suffix) or len(y_suffix):
        data = _select(data, x_suffix, y_suffix)
    return data


def unpack_batches(x: np.ndarray, padding: int = 0):
    """Reshape a batch-major tensor to a time-major tensor."""

    if padding > 0:
        x = x[:, padding:-padding, ...]
    x = x.reshape((-1, x.shape[-1]))
    return x


def get_data_from_gen(data_gen):
    x, y = data_gen.unroll(return_x=True, merge_batches=True)
    x = unpack_batches(x, data_gen.data_padding)
    if y is not None:
        y = unpack_batches(y, data_gen.data_padding)
    return x, y


def sub_range(data_len, fraction: float, min_nb_samples: int = 0, seed=None):
    """Select a contiguous random subset of a dataset."""

    np.random.seed(seed)
    sub_len = int(max(np.ceil(fraction * data_len), np.ceil(min_nb_samples)))
    first_sample = np.random.randint(low=0, high=data_len - sub_len - 1)
    last_sample = first_sample + sub_len + 1
    return first_sample, last_sample


def compute_class_weights(y: np.ndarray) -> List[float]:
    """Compute inverse-frequency class weights over a chunked label array."""

    nb_classes = y.shape[1]

    yy = dask.array.from_array(y)
    nb_chunks = len(yy.chunks[0])

    counts = np.zeros((nb_chunks, nb_classes))
    for cnt, block in enumerate(tqdm(yy.blocks, total=nb_chunks, desc="Counting class occurrences")):
        counts[cnt, :] = np.sum(block.compute().astype(float), axis=0)

    class_weights = np.sum(counts, axis=0)
    class_weights /= np.sum(class_weights)
    class_weights = [1 / class_weight for class_weight in class_weights]
    return class_weights


class AudioSequence(keras.utils.Sequence):
    """Keras sequence backed by arrays or memory-mapped datasets."""

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        nb_hist: int = 1,
        y_offset: Optional[int] = None,
        stride: int = 1,
        cut_trailing_dim: bool = False,
        with_y_hist: bool = False,
        data_padding: int = 0,
        first_sample: int = 0,
        last_sample: Optional[int] = None,
        output_stride: int = 1,
        nb_repeats: int = 1,
        shuffle_subset: Optional[float] = None,
        unpack_channels: bool = False,
        mask_input: Optional[int] = None,
        batch_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        class_weights: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        self.x, self.y = x, y

        self.first_sample = first_sample
        self.last_sample = self.x.shape[0] if last_sample is None else last_sample
        self.nb_samples = self.last_sample - self.first_sample
        self.nb_repeats = nb_repeats
        self.output_stride = output_stride
        self.with_y = self.y is not None
        if self.with_y:
            self.nb_classes = self.y.shape[-1]
        else:
            self.nb_classes = 0

        self.batch_size = batch_size
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_subset = shuffle_subset
        self.x_hist = nb_hist
        self.with_y_hist = with_y_hist
        self.data_padding = data_padding
        self.unpack_channels = unpack_channels
        self.class_weights = class_weights
        self.mask_input = mask_input
        s0 = self.first_sample / self.stride
        s1 = (self.last_sample - self.x_hist - 1) / self.stride
        self.allowed_batches = np.arange(s0, s1, dtype=np.uintp)
        if self.shuffle_subset is not None:
            self.allowed_batches = np.random.choice(
                self.allowed_batches, size=int(len(self.allowed_batches) * self.shuffle_subset), replace=False
            )

        if y_offset is None:
            self.y_offset = int(self.x_hist / 2)
        else:
            self.y_offset = int(y_offset)

        if self.with_y_hist:
            self.weights = np.ones((self.batch_size, self.x_hist))
            if self.data_padding > 0:
                self.weights[:, : self.data_padding] = 0
                self.weights[:, -self.data_padding :] = 0
            self.weights = self.weights[:, ::output_stride]
        else:
            self.weights = np.ones((self.batch_size,))

        self.batch_processor = batch_processor
        self._idx_offset = 0

    def unroll(self, return_x=True, merge_batches=True):
        xx = None
        if return_x:
            xx = np.zeros((len(self), self.batch_size, self.x_hist, *self.x.shape[1:]))
        if self.with_y_hist:
            yy = (
                np.zeros((len(self), self.batch_size, int(self.x_hist / self.output_stride), self.nb_classes))
                if self.with_y
                else None
            )
        else:
            yy = np.zeros((len(self), self.batch_size, self.nb_classes)) if self.with_y else None

        for cnt, gen_output in enumerate(self):
            if return_x:
                if self.unpack_channels:
                    xx[cnt, ...] = np.concatenate(gen_output[0], axis=-1)
                else:
                    xx[cnt, ...] = gen_output[0]
            if self.with_y:
                yy[cnt, ...] = gen_output[1]

        if merge_batches:
            if return_x:
                xx = xx.reshape((len(self) * self.batch_size, self.x_hist, *self.x.shape[1:]))
            if self.with_y_hist:
                yy = (
                    yy.reshape((len(self) * self.batch_size, int(self.x_hist / self.output_stride), self.nb_classes))
                    if self.with_y
                    else None
                )
            else:
                yy = yy.reshape((len(self) * self.batch_size, self.nb_classes)) if self.with_y else None

        if self.with_y:
            out = (xx, yy)
        else:
            out = (xx,)
        return out

    def __len__(self):
        return int(
            self.nb_repeats
            * max(
                0,
                np.floor(
                    (self.nb_samples - ((self.stride * (self.batch_size - 1)) + self.x_hist))
                    / (self.stride * self.batch_size)
                )
                + 1,
            )
        )

    def __str__(self):
        string = [
            "AudioSequence with {} batches each with {} items.\n".format(len(self), self.batch_size),
            "   Total of {} samples with\n".format(self.nb_samples),
            "   each x={} and\n".format(self.x.shape[1:]),
        ]
        string.append("   each y={}".format(self.y.shape[1:])) if self.y is not None else "no y."
        return "".join(string)

    def __getitem__(self, idx):
        idx += self._idx_offset
        batch_x = np.zeros((self.batch_size, self.x_hist, *self.x.shape[1:]), dtype=self.x.dtype)

        if self.with_y:
            if self.with_y_hist:
                batch_y = np.zeros(
                    (self.batch_size, int(self.x_hist / self.output_stride), self.nb_classes), dtype=self.y.dtype
                )
            else:
                batch_y = np.zeros((self.batch_size, self.nb_classes), dtype=self.y.dtype)

        if self.shuffle:
            pts = np.random.choice(self.allowed_batches, size=self.batch_size, replace=False)
        else:
            pts = range(
                int(self.first_sample / self.stride) + idx * self.batch_size,
                int(self.first_sample / self.stride) + (idx + 1) * self.batch_size,
            )

        for cnt, bat in enumerate(pts):
            batch_x[cnt, ...] = self.x[int(bat * self.stride) : int(bat * self.stride + self.x_hist), ...].copy()

            if self.with_y:
                if self.with_y_hist:
                    batch_y[cnt, ...] = self.y[
                        int(bat * self.stride) : int(bat * self.stride + self.x_hist) : self.output_stride, ...
                    ]
                else:
                    batch_y[cnt, ...] = self.y[int(bat * self.stride + self.y_offset), ...]
        if self.unpack_channels:
            batch_x = [batch_x[..., chn][..., np.newaxis] for chn in range(batch_x.shape[-1])]

        if self.mask_input is not None:
            batch_x[:, int(batch_x.shape[1] / 2 - self.mask_input) : int(batch_x.shape[1] / 2 + self.mask_input), :] = 0

        if self.batch_processor is not None:
            batch_x = self.batch_processor(batch_x)

        if self.with_y:
            out = (batch_x, batch_y)

            if self.class_weights is not None:
                weights = np.zeros_like(self.weights)
                labels = np.argmax(batch_y, axis=-1)
                for label, weight in enumerate(self.class_weights):
                    weights[labels == label] = weight
                weights *= self.weights
            else:
                weights = self.weights

            if self.data_padding > 0:
                out = (batch_x, batch_y, weights)
        else:
            out = (batch_x,)
        out = [o.astype(np.float32) for o in out]
        return out
