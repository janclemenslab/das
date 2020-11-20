# Data formats

## Exported annotations and audio
Produced inn the GUI via `File/Save annotations` and `File/Export for DeepSS`.

Audio and annotations are exported into `csv` (comma-separated values) and `npz` (zip compressed numpy files):
- `npz` consist of two variables:
    + `data`: `[samples, channels]` array with the audio data
    + `samplerate`: `[1,]` array with the sample rate in Hz
- `csv` contains three columns:
    + `name` - the name of the song or syllable type
    + `start_seconds` - the start time of the syllable.
    + `stop_seconds` - the stop of the syllable. Start and stop are identical for song types of type event, like the pulses of fly song.
    + Each row in the file contains to a single annotation with `name`, `start_seconds` and `stop_seconds`. Special rows a reserved for song types without any annotations: For syllables or other segment types, the consist of the name, `start_seconds` is `np.nan` and an arbitrary stop_seconds. For event-like types (song pulses), both `start_seconds` and `stop_seconds` are `np.nan`.

## Dataset for training
For training, _DeepSS_ expect data in a specific, dictionary-like format. This format can be implemented using multiple backends, such as python's `dict`, `h5py`, `zarr`, or our own fast `dss.npy_dir`. In the following, we explain how to create the structure using the graphical user phase and the python module, and describe the expected data structure for users that want to implement it themselves.

Data is organized in a dictionary-like (mappable) structure with the data splits for training, validation, and test stored in top-level keys `train`, `val`, and `test`. Training and validation data are required. Validation is used during training to monitor progress and adjust the learning rate or stop training early. Test data is optional - it is used after training to assess the performance of the model.

At the top-level, metadata is stored in an `attrs` attribute with the following fields
Each top-level key
Such a structure can be implemented in many different ways - using python's `dict`, `h5py`, `zarr`, or `dss.npy_dir`.

Each a dict with the following keys:

- Inputs `x`: `[samples, channels]`
    - First dim is always samples, last is typically audio channels (>=1).
    - For spectrogram representations, could also be frequency channels `[samples, freqs]` for single-channel recordings. For multi-channel data time, the frequency channels from the spectrum of each audio channels can be stacked to `[time, channels*freqs]`
- Targets `y` for each sample in 'x': `[samples, nb_classes]`
    - Binary (0/1) or a probability but should sum to 1.0 across classes for each sample
    - Multiple targets can be specified by adding target variables with a suffix `y_somesuffix`. That allows you to train the a network with the same x but different y's (e.g. pulse or sine-only). *Must* named like `y_somesuffix`, where `somesuffix` can be an arbitrary string. The target can be specified during training with the `y_suffix=somesuffix` argument, which defaults to '' (the standard target w/o suffix `y`). Training y-suffixes also requires suffix-specific attributes specifying the name and type of the target - `classnames_somesuffix` and `classtype_somesuffix`.
- Metadata in dict `data.attrs`:
    - `samplerate_x_Hz`, `samplerate_y_Hz` - should be equal
    - `samplerate_song_Hz` (optional) - of `song`, the original recording in case `x` is the result of a wavelet or spectrogram transform
    - class name and type information for each target:
        + `classnames`: `[nb_classes: str]` - one for each 2nd dim in y
        + `classtype`: `[nb_classes: str]` (optional for training) - `event` (e.g. pulse) or `segment` (sine or syllable)
        + for each target specified via `y_somesuffix`, create a `classnames_suffix` and `classnames_somesuffix` attribute

Data is accessed via `data['train']['x']` and the metadata via `data.attrs`. `attrs` is a standard attribute for storing metadata in `hdf5` and `zarr` files and can be easily attached to a standard dictionary: `a = dict(); a.attrs = {'samplerate_x_Hz': 10_000}`.

 During training, data is loaded  (see `dss.io`) with the correct backend given by the extension of the file or directory. Currently supported files types are:
 - [zarr](https://zarr.readthedocs.io/) storages (files or directories ending in `.zarr`),
 - [hdf5](http://docs.h5py.org/) files (files ending in `.h5`),
 - directories ending in `.npy` with [numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html) files (see [below](#npy-dir-format)).

`zarr` is great for assembling and storing datasets, in particular for datasets with unknown final size since it allows appending to arrays, and is used as an intermediate storage during dataset assembly. For training, `npy` is the fastest, in particular for random access of datasets that are too large to fit in memory, thanks to highly efficient memory mapping.


## `npy_dir` format
We provide a alternative storage backend - `npy_dir` ([source](../src/dss/npy_dir.py)) - that mirrors the data structure in directory hierarchy with [numpy's npy](https://numpy.org/doc/stable/reference/generated/numpy.load.html) files (inspired by Cyrille Rossant's series of blog posts ([1](https://cyrille.rossant.net/moving-away-hdf5/), [2](https://cyrille.rossant.net/should-you-use-hdf5/)), [jbof](https://github.com/bastibe/jbof) and [exdir](https://exdir.readthedocs.io/)). Keys map to directories; values and `attrs` map to `npy` files (see `dss.npy_dir`). For instance, `data['train']['x']` is stored in `dirname.npy/train/x.npy`. `attrs` is stored in the top-level directory.

This provides structured access to data via npy files. npy files have the advantage of providing a fast memory-mapping mechanism for out-of-memory access if your data set does not fit in memory. While zarr, h5py, and xarray provide mechanisms for out-of-memory access, they tend to be generally slower or require fine tuning to reach the performance reached with memmapped npy files.

