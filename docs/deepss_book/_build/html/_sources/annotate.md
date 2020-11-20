# Annotate song manually
Manual annotations are required for training a _DeepSS_ to annotate your recordings. If already have annotated data, head straight to [make a dataset for training](make_dataset).

Otherwise, we provide a GUI for annotating song manually.
Start the GUI by opening a terminal, activating the conda environment, and typing `xb`:
```shell
conda activate dss
xb
```
This opens a window with options for loading data - `Load audio from file`:

<img src="images/xb_start.png" alt="start screen" width=450>


## Loading audio data
This will open a dialog for selecting a file. The GUI can load audio data in a wide range of data formats:

- audio files (readable by [soundfile](http://pysoundfile.readthedocs.io/), [list of supported formats](http://www.mega-nerd.com/libsndfile/#Features))
- hdfs files ([h5py](http://docs.h5py.org/)). Matlab's `.mat` [files](https://www.mathworks.com/help/matlab/ref/save.html#btox10b-1-version) also use this format.
- numpy's `npy` or `npz` files ([numpy](https://numpy.org/doc/stable/reference/routines.io.html))

```{note}
If your favorite format is not included in this list, try to convert it to a `wav` file.
```
After selecting a file, a menu allows you to fine tune things:

- which dataset
- data format - xb infers, you can correct
- sample rate - xb infers, but some formats do not have that information so you may need to correct or provide
- ignore_tracks (REMOVE)
- crop (REMOVE)
- annotations: This will also attempt to load existing annotations - it will default to the selected filename with the extension replaced by `.csv` but an alternative can be selected. The annotations need to be saved as comma-separated values with a header naming three columns: name, start_seconds, stop_seconds and each following row containing the name of the song type and the start and stop of the song in the recording in seconds. For events, start and stop need to have identical values. See [technical specification](data_formats.html#exported-annotations-and-audio))
- initialize annotations: “name,category;name2,category2” (category is either event or segment)
- Sample rate events (REMOVE)
- the range of frequencies displayed in the spectrogram views to focus the display on the frequencies that occur in the song
- band-pass filter the audio to remove noise
- cue points

:::{figure} xb_load-fig
<img src="images/xb_load.png" alt="loading screen">

Loading screen.
:::


Most of these parameters can also be specified using the xb command line interface (see [xb_cli]).

## Overview over the display and menus
Audio data from all channels (gray), with one channel being selected (white), and the spectogram of the currently selected channel below. Move forward/backward along the time axis via the A/D keys and zoom in/out the time axis with W/S keys (See Playback/). The temporal resolution of the spectrogram can be increased at the expense of frequency resolution with the R and T keys.


:::{figure} xb_display-fig
<img src="images/xb_display.png" alt="waveform and spectrogram display" width="100%">

Waveform (top) and spectrogram (bottom) display of a multi-channel recording.
:::

You can choose to hide all non-selected channels in the waveform view by toggling “Audio/Show all channels”. To change the channel for which the spectrogram is displayed, use the dropdown list on the upper right or switch to next/previous channel with the up/down arrow keys. Pressing `Q` (Audio/Autoselect loudest channel) will toggle automatically selecting the loudest channel in the current view.

<!-- :::{figure} markdown-fig
<img src="images/xb_channels.png" alt="select channels" width="100%">

Channel selection
::: -->



## Annotate song
Song types for annotation are taken from existing annotations if they were loaded. Song types for annotation can be defined upon load (see above). You can also add new or edit the names of existing song types via “Audio/Add or edit annotation types”.

:::{figure} xb_make-fig
<img src="images/xb_make.png" alt="edit annotation types" height="500px">

Create, rename or delete song types for annotation.
:::


The dropdown many on the top left of the waveform view can be used to select which song types you want to annotate. The selected annotation type can also be changed with number keys according to the number indicated in the dropdown menu.

:::{figure} xb_types-fig
<img src="images/xb_types.png" alt="select annotation types" height="150px">

Select annotation types.
:::

To annotate a new song, left-click on the waveform or spectrogram view. If you have selected an event-like type, that’s it - you just placed the event time and a line should appear in the waveform and spectrogram view. If you have selected a segment type, the cursor changes to a cross. You have only placed one boundary of the segment, a second click somewhere else will complete the annotation and a shaded area should appear between the time points of the first and second click.


:::{figure} xb_annotate-fig
<img src="images/xb_annotate.png" alt="annotate song" height="500px">

Fully annotated fly song. Pulse (cyan) is an event-like song type and is annotated by clicking on the pulse center. Sine (red shaded area) is a segment- or syllable-like song type and is annotated by clicking on the beginning and the end of the sine syllable.
:::

Delete annotations by right-clicking on the annotation. You can delete all annotations or only the selected annotation with the U and Y, respectively or via the Audio menu. Move annotations by dragging the lines or the boundaries of the shaded area - this will change event times and segment bounds. Or drag the shaded area itself to move the whole segment. Movement can be disabled completely or restricted to the currently selected annotation type.

## Save annotations
Save the annotations via the file menu as a comma-separated `csv` file. See [here](data_formats.html#exported-annotations-and-audio) for a specification of the file format.
