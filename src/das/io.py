"""
Load data for training/testing.
See doc/data.md for a description of the data schema.
"""
import os.path
from . import npy_dir


def _select(data, x_suffix, y_suffix):
    for lvl in ['test', 'val', 'train']:
        if lvl in data:
            if 'y_' + y_suffix in data[lvl]:
                data[lvl]['y'] = data[lvl]['y_' + y_suffix]
                if 'eventtimes_' + y_suffix in data[lvl]:
                    data[lvl]['eventtimes'] = data[lvl]['eventtimes_' + y_suffix]
            if 'x_' + x_suffix in data[lvl]:
                data[lvl]['x'] = data[lvl]['x_' + x_suffix]
                if 'eventtimes_' + x_suffix in data[lvl]:
                    data[lvl]['eventtimes'] = data[lvl]['eventtimes_' + x_suffix]

    if f'samplerate_x_{x_suffix}_Hz' in data.attrs:
        data.attrs['samplerate_x_Hz'] = data.attrs[f'samplerate_x_{x_suffix}_Hz']

    if 'class_names_' + y_suffix in data.attrs and 'class_types_' + y_suffix in data.attrs:
        data.attrs['class_names'] = data.attrs['class_names_' + y_suffix]
        data.attrs['class_types'] = data.attrs['class_types_' + y_suffix]
    return data


def _to_dict(data):
    "Convert dict-like zarr or h5 store `data` to python dictionary."
    d = npy_dir.DictClass()
    d.attrs = dict(data.attrs)  # cast to dict since data.attrs are read-only for zarr stores
    for key_top in data.keys():
        d[key_top] = dict()
        for key, val in data[key_top].items():
            d[key_top][key] = val
    return d


def load(location, x_suffix='', y_suffix=''):
    """Load data for training/testing from zarr store, npy directory, or hdf5 file.

    Args:
        location ([type]): [description]
        x_suffix, y_suffix (str, optional): alternative key for the training source and target (allows for different x/y's for the same y/x in one data file)
    Returns:
        dict-like complying with data schema defined above
    """

    location = os.path.normpath(location)  # remove trailing path separators
    if location.endswith('.zarr'):
        import zarr
        data = zarr.open(location, mode='r')
    elif location.endswith('.h5'):
        import h5py
        data = h5py.File.open(location, mode='r')
    elif location.endswith('.npy'):
        data = npy_dir.load(location)
    else:
        raise ValueError(f'Could not load data. Location {location} has unknown extension - needs to end either in ".zarr", ".npy", or ".h5".')

    data = _to_dict(data)

    if len(x_suffix) or len(y_suffix):
        data = _select(data, x_suffix, y_suffix)
    return data
