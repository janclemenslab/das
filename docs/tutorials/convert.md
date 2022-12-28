# Convert your own annotations and audio data
If you start from scratch---with non-annotated audio recording---use the GUI for labelling the data. See the [GUI tutorial](/tutorials_gui/tutorials_gui) for a description of all steps - from loading data, annotating song, making a dataset, training a network and generating annotations after training.

However, often annotations exist, from old manual annotations or other tools. You can use existing annotations to train _DAS_, by converting the existing annotations into the _DAS_ format. See [here](/technical/data_formats).

If audio data is in a format supported by das (see [here](/tutorials_gui/load)), open in GUI and export to a folder. For processing large sets of recordings use the [notebook](make_ds_notebook).

<!--
## Format of exported annotations and audio
Produced by the GUI via `File/Save annotations` and `File/Export for DAS`.

Audio and annotations are exported into `csv` (comma-separated values) and `npz` (zip compressed numpy files):
- `npz` consist of two variables:
    + `data`: `[samples, channels]` array with the audio data
    + `samplerate`: `[1,]` array with the sample rate in Hz
- `csv` contains three columns:
    + `name` - the name of the song or syllable type
    + `start_seconds` - the start time of the syllable.
    + `stop_seconds` - the stop of the syllable. Start and stop are identical for song types of type event, like the pulses of fly song.
    + Each row in the file contains to a single annotation with `name`, `start_seconds` and `stop_seconds`. Special rows a reserved for song types without any annotations: For syllables or other segment types, the consist of the name, `start_seconds` is `np.nan` and an arbitrary stop_seconds. For event-like types (song pulses), both `start_seconds` and `stop_seconds` are `np.nan`. -->


<!--
## Annotation format
csv file with three columns:
- `name`: name of the song element for instance 'pulse' or 'sine' or 'syllable A'
- `start_seconds`: *start* of the song element in seconds rel. to the start of the recording
- `stop_seconds`: *end* of the song element in seconds rel. to the start of the recording

There are two types of song elements:
- `events` have not extent in time, `start_seconds=stop_seconds`, and are best used for brief, pulsatile signals like fly pulse song
- `segments` extend in time, `start_seconds>stop_seconds`, and should be used for normal syllables or fly sine song -->

<!-- The `csv` format is universal and can be created and edit using Excel or even a plain text editor. It is also very easy to programmatically created `csv` files from your own annotation format in python using a [pandas DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html). Below, we show an example of creating a DataFrame in the correct format from annotation data and saving it as a `csv` file for use with _DAS_.


### Examples of transforming custom annotation formats
Say we have read annotation data into python as three lists containing the names, start and stop of a song type:
```python
names = ['bip', 'bop', 'bip']
start_seconds = [1.34, 5.67, 9.13]
stop_seconds = [1.34, 5.85, 9.13]
```
This defines two song types, "bip" and "bop". "bip" an event-like song type (like a pulse in fly song), since start and stop are identical. "bop" is a segment-like song type (like a syllable in birdsong), because start and stop differ.

This information needs to be arranged into a table, i.e., a pandas DataFrame, of this format:
| names   | start_seconds    | stop_seconds  |
| ------- |:-------------|:-----|
| bip      | 1.34 | 1.34 |
| bop      | 5.67   | 5.85 |
| bip      | 9.13   |  9.13 |

A pandas DataFrame with the format can be created by two means:
Use `xb.annot`, which takes the three lists and produces a correctly  DataFrame
```python
from xarray_behave.annot import Events  # require install with gui

evt = Events.from_lists(names, start_seconds, stop_seconds)
df = evt.to_df()
```

Or you can assemble the DataFrame yourself:

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

Then save as a `csv` file:
```python
df.to_csv('filename.csv')
```


### Convert audio data
The GUI can read many formats (see [list of supported audio formats](/tutorials_gui/load)) and data can always be exported in the correct format via the GUI.
However, if you want to assemble many of your own recordings into dataset for training, a programmatic approach is more efficient.

To assemble a dataset, audio data has to be provided in two formats:
- `wav`: Universal format for audio data. can be created from many software packages:
    - From python `scipy.io.wavfile.write(...)`
    - from matlab `wavwrite(...)`
    - from the command line via ffmpeg: `ffmpeg ...`
- `npz`: Python-specific but a bit more flexible/robust. Should contain two variables - `samplerate` and `data` - and can be created like so: `np.savez(filename, data=audio, samplerate=samplerate)` -->

<!--
```{warning}
Clipping can occur when saving certain data types as wav files. see docs of [scipy.io.wavfile.write](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html) for a list of the range of values available when saving audio of different types to wav.
``` -->