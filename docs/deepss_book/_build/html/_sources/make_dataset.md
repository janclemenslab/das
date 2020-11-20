# Making a data set for training
Inference on any data but training expects data in a specific format with splits and metadata (see [specification](data_formats.html#dataset-for-training)).
Three(ways):
1. From manual annotations created in the `xb` GUI
2. From your own annotations

We provide tools that more or less automate the generation of these datasets from audio annotated using the GUI but also present sample code for using your own data formats.

## Using the GUI
Once you have annotated a recording, export it for DeepSS (File/Export for DeepSS). This will save the annotations as csv. You can export a specific song type or all types.

Audio (and sample rate information) can be saved as WAV or NPZ (compressed numpy binary file). We recommend NPZ, because it is robust and portable. WAV is more general but the format is more restricted and can lead to data loss. For instance, floating point data is restricted to the range [-1, 1]. You can compensate for this by setting the re-scaling the data via s scale factor to avoid clipping when saving.

Export only a selected range - for fast training - annotate 5 seconds, export only the annotated part. Select the range to ensure that all three splits (train/validate/test) will end up with annotated song. For instance, the beginning and end should contain annotated - do not export a 15 second snippet where the first 10 seconds are silent - during splitting, this will result in some splits - for instance the validation set - being without annotations and can result in poor training results.

Annotate and export multiple recordings into the same folder and then assemble a dataset via DeepSS/Make dataset.

By default, the dataset will be assembled to allow training a single network that recognizes all annotated song types. If you want to train network for specific song types, then make training target for individual song types. For instance, fly song typically consists of sine and pulse song. We found training a pulse and a separate sine networks improves accuracy.

Next set how annotation types are encoded in the training target. Events are marked by a single time point - to make training more robust, events should be represented by gaussian with a standard deviation. To simplify post processing of segments, in particular for birdsong with its many syllable types that are directly adjacent, we found that introducing brief gaps helps with post-processing the inferred annotations.

Lastly, the data is split into three sets, which are used during different phases of the training:
- _train_: optimize the network parameters
- _validation_: monitor network performance during training and steer training (stop early or adjust learning rates)
- _test_: independent data to assess network performance after training)

Data can by split based on files or based on samples:
-  _Split by files_: select a fraction of files for the specific split. The full file will be used.
- _Split by samples_: select a fraction of data from each file.

If you have enough files, split by validation and ideally test by files - different files may come from different individuals so you test and validate the network based on how well it generalizes across individuals. If you have too few files or the different song types do not appear in all files, split by samples.

Inspect the data set with the [1_inspect_data.ipynb]() notebook


To learn how to annotate song, DeepSS requires annotated audio data. The annotated audio is split into three parts:

- `train`: The inference of Deeps is compared to the provided annotations for this part and errors are used to update the DeepSS network during training.
- `val` (aka validation): Track network performance during training and decide when to save the network, adjust the training strategy or stop training.
- `test`: Assess the quality of the DeepSS inference after training.

Each of the three parts contains two main variables:

- `x`: The audio data in the form `[samples x channels]`. Can be pre-processed, for instance filtered for noise.
- `y`: The annotations in the form `[samples x song types]`, encoded as the probability of finding each song type (or noise) at each sample. These are the targets that DeepSS is optimized to reproduce during training.
- `y_suffix`: Alternative annotations for the audio in `x` can be provided here. For instance, if you want to train DeepSS to detect only one of the song types in the data, you can make an annotation trace `y_specificsongtype` and provide the `y_suffix=specificsongtype` as an argument during training.
- metadata:
    - the sample rate of `x` and `y` in Hz.
    - `class_names` and `class_types` (event or segment DEFINE)
    - `class_names_suffix` and `class_types_suffix` (for each suffix - would be ['noise', 'suffix'])


DeepSS expects this information to come in a specific format, see Section 3 below and [data_schema.md] for details. Briefly, the data should be provided in a nested dict-like structure (or mappable):

There are three ways of providing data for training, with increasing levels of complexity and flexibility:

1. (simplest, least flexible) Folder with the data and annotations in a specific format (wav and csv files).
2. (intermediate simplicity and flexibility) Folder with the data and annotations in a custom format - requires providing custom functions for loading both data types.
3. (complex, maximally flexible) Bring your own mappable.

## Using your own data
See [tutorials/1_prepare_data.ipynb]() for an example. This allows you more flexibility and to use your own data formats.



## 1. Folder with wav and csv
`data` folder with `*.wav` files with the recording and matching `*.csv` files with the annotations - recordings and annotations will be matched according to the file base name:
```shell
data\
    file01.wav
    file01.csv
    another_file.wav
    another_file.csv
    yaf.wav
    yaf.csv
```
### Annotation format
csv file with three columns:
- `name`: name of the song element for instance 'pulse' or 'sine' or 'syllable A'
- `start_seconds`: *start* of the song element in seconds rel. to the start of the recording
- `stop_seconds`: *end* of the song element in seconds rel. to the start of the recording

There are two types of song elements:
- `events` have not extent in time, `start_seconds=stop_seconds`, and are best used for brief, pulsatile signals like fly pulse song
- `segments` extend in time, `start_seconds>stop_seconds`, and should be used for normal syllables or fly sine song

### Recording format
Single or multi channel wave file (should be readable via `scipy.io.wavefile.read`).

## 2. Custom loaders for the recordings and annotations
Same general data structure as above but with custom data formats. If your recordings and annotations are not in the format expected by the standard loaders used above (`scipy.io.wavefile.read` for the recordings, `pd.read_csv` with name/start_seconds/stop_seconds for annotations), or if it's hard to convert your data into these standard formats, you can provide your own loaders as long as they conform to the following interface:

- _data loaders_: `samplerate, data = data_loader(filename)`, accepts a single string argument - the path to the data file and returns two things: the samplerate of the data and a numpy array with the recording data [time, channels]. Note: `scipy.io.wavefile.read` returns `[time,]` arrays - you need to add a new axis to make it 2d!
- _annotation loaders_: `df = annotation_loader(filename)`, accepts a single string argument with the file path and returns a pandas DataFrame with these three columns: `name`, `start_seconds`, `stop_seconds` (see 1).

ref notebook

#### Example 1: More exotic audio files
You can use [audiofile](https://pypi.org/project/audiofile/) for reading audio data from more exotic formats. Just be aware that the order of outputs is not as required - the first return argument of `audiofile.read` is the data, the second one the sample rate:
```python
import audiofile as af
data, samplerate = af.read('signal.aif')
```
A simple wrapper that reverses the order of the output will do
```python
import audiofile as af
def data_loader(filename):
    data, samplerate = af.read(filename)
    return samplerate, data
```

#### Example 2: Audio data in HDF5 or matlab files
If you data is saved in HDF5 format, with the recording in a 'samples' field and the sample rate as an attribute, you could use the following wrapper:
```python
import h5py
def data_loader(filename):
    with h5py.File(filename, 'r') as f:
        data = f['samples'][:]
        samplerate = f.attrs['samplerate']
    return samplerate, data
```
If sample rate is not saved with the data, you can return a constant value: `return 10_000, data`.
Recent versions of matlab also save data as HDF5 so this should work for `mat` files. Otherwise use `scipy.io.loadmat`

### Example 3: Transforming a custom annotation format
Build DataFrame from segment on- and offsets and event times loaded from a custom format:
```python
import numpy as np
import pandas as pd

# create empty DataFrame with the required columns
df = pd.DataFrame(columns=['name', 'start_seconds', 'stop_seconds'])

# append a segment
onset = 1.33 # seconds
offset = 1.42  # seconds
segment_bounds = [onset, offset]
segment_name = 'sine_song'

new_row = pd.DataFrame(np.array([segment_name, *segment_bounds])[np.newaxis,:],
                        columns=df.columns)
df = df.append(new_row, ignore_index=True)

# append an event
event_time = 2.15 # seconds
event_name = 'pulse'

new_row = pd.DataFrame(np.array([event_name, event_time, event_time])[np.newaxis,:],
                        columns=df.columns)
df = df.append(new_row, ignore_index=True)
```


## 3. Bring your own mappable
DeepSS expects a simple dictionary-like data structure (see npy_dir doc):
```
data
  ├── ['train']
  │   ├── ['x']         (the audio data - samples x channels)
  │   ├── ['y']        (annotations - samples x song types, first one is noise, needs to add to )
  │   ├── ['y_suffix1'] (optional, multiple allowed)
  ├── ['val']
  │   ├── ['x']
  │   ├── ['y']
  │   ├── ['y_suffix1']
  ├── ['test']
  │   ├── ['x']
  │   ├── ['y']
  │   ├── ['y_suffix1']
  └── attrs
        └── ['samplerate'] (of x and y in Hz)
              ['class_names']
              ['class_types'] (event or segment)
              ['class_names_suffix1']
              ['class_types_suffix1'] (event or segment)
```

Data is accessed via keys, for instance `data['train']['x']`. `attrs` is a dictionary accessed via `.` notation: `data.attrs['samplerate']`.

This structure can be implemented via python's builtin [dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries), [hdf5](https://www.h5py.org), [xarray](http://xarray.pydata.org'), [zarr](https://zarr.readthedocs.io), or anything else that implements a key-value interface (called a Mapping in python).

We provide a alternative storage backend - `npy_dir` ([source](../src/dss/npy_dir.py)) - that mirrors the data structure in directory hierarchy with [numpy's npy](https://numpy.org/doc/stable/reference/generated/numpy.load.html) files (inspired by Cyrille Rossant's series of blog posts ([1](https://cyrille.rossant.net/moving-away-hdf5/), [2](https://cyrille.rossant.net/should-you-use-hdf5/)), [jbof](https://github.com/bastibe/jbof) and [exdir](https://exdir.readthedocs.io/)). For instance, `data['train']['x']` is stored in `dirname/train/x.npy`. `attrs` is stored as a `yaml` file in the top directory.

This provides structured access to data via npy files. npy files have the advantage of providing a fast memory-mapping mechanism for out-of-memory access if your data set does not fit in memory. While zarr, h5py, and xarray provide mechanisms for out-of-memory access, they tend to be generally slower or require fine tuning to reach the performance reached with memmapped npy files.

_Notes_

- that the annotations correspond to probabilities - they should sum to 1.0 for each sample. The first "song" type, should be noise or no song, `p(no song)`
- y_suffix...