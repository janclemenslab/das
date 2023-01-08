# Quick start tutorial (bird)
This quick start tutorial walks through all steps required to make _DAS_ work with your data, using a recording of zebra finch song as an example. A comprehensive documentation of all menus and options can be found in the [GUI documentation](/tutorials_gui/tutorials_gui).

In the tutorial, we will train _DAS_ using an iterative and adaptive protocol that allows to quickly create a large dataset of annotations: Annotate a few syllable renditions, fast-train a network on those annotations, and then use that network to predict new annotations on a larger part of the recording. These first predictions require manually correction, but correcting is typically much faster than annotating everything from scratch. This correct-train-predict cycle is then repeated with ever larger datasets until network performance is satisfactory.

## Download example data
To follow the tutorial, download and open these two audio files:

- [birdname_130519_113032.2.wav](https://github.com/janclemenslab/DAS/releases/download/data/birdname_130519_113032.2.wav). We will use this file for training a DAS network.
- [birdname_130519_110831.1.wav](https://github.com/janclemenslab/DAS/releases/download/data/birdname_130519_110831.1.wav). We will use this file for testing the trained DAS network.

The recordings are of a Zebra finch male, recorded by Jack Goffinet et al. (part of [this dataset](https://research.repository.duke.edu/concern/datasets/9k41zf38g)). We will walk through loading, annotating, training and predicting using these files as examples.

## Start the GUI

Install _DAS_ following these [instructions](/installation). Then start the GUI by opening a terminal, activating the conda environment created during install and typing `das gui`:
```shell
conda activate das
das gui
```

The following window should open:


:::{figure-md} xb_start-fig
<img src="images/xb_start.png" alt="start screen" width=450>

Start screen.
:::


## Load audio data

Choose _Load audio from file_ and select the downloaded recording of fly song.

In the dialog that opens, leave everything as is except.

:::{figure-md} xb_load-fig
<img src="images/xb_quick_bird_load.png" alt="loading screen">

Loading screen.
:::


## Waveform and spectrogram display
Loading the audio will open a window that displays the first second of audio as a waveform (top) and a spectrogram (bottom).

To navigate the view: Move forward/backward along the time axis via the `A`/`D` keys and zoom in/out the time axis with the `W`/`S` keys (see also the _Playback_ menu). You can also navigate using the scroll bar below the spectrogram display or jump to specific time points using the text field to the right of the scroll bar. The temporal and frequency resolution of the spectrogram can be adjusted with the `R` and `T` keys.

You can play back the waveform on display through your headphones/speakers by pressing `E`.

:::{figure-md} xb_display-fig
<img src="images/xb_quick_bird_view.png" alt="waveform and spectrogram display" width="100%">

Waveform (top) and spectrogram (bottom) display of fly song.
:::


## Initialize or edit syllable
Before you can annotate song, you need to register the different syllable of the main motif. This bird's motif consists of six syllables.

Add the six syllables for annotation via the _Add/edit_ button at the top of the windows or via the _Annotations/Add or edit song types_ menu. Let's name them `syll1` to `syll6`:

:::{figure-md} xb_make-fig
<img src="images/xb_quick_bird_make.png" alt="edit annotation types" height="400px">

Create new syllables for annotation.
:::


## Create annotations manually
The six syllables can now be activated for annotation using the dropdown menu on the top left of the window. The active syllable can also be changed with the number keys indicated in the dropdown menu---in this case `1`...`6`.

Song is annotated by left-clicking the waveform or spectrogram view.Annotating a syllable requires two clicks---one for the onset and one for the offset of the syllable.

:::{figure-md} xb_create-fig
<img src="/images/xb_bird_create.gif" alt="annotate song" width="700px">

Left click on waveform or spectrogram view to create annotations.
:::

## Edit annotations
In case you misclicked, you can edit and delete annotations. Edit  syllable bounds by dragging the boundaries of segments. Drag the shaded area itself to move a syllable without changing its duration. Movement can be disabled completely or restricted to the currently selected annotation type via the _Annotations_ menu.

Delete annotations of the active syllable by right-clicking on the annotation. Annotations of all syllable types or of only the active one in the view can be deleted with `U` and `Y`, respectively, or via the _Annotations_ menu.

:::{figure-md} xb_edit-fig
<img src="images/xb_bird_edit.gif" alt="annotate song" width="700px">

Dragging moves, right click deletes annotations.
:::

Change the label of an annotation via CMD/CTRL+Left click on an existing annotation. The type of the annotation will change to the currently active one.


## Export annotations and make a dataset
_DAS_ achieves good performance from few annotated examples. Once you have completely annotated the six syllables in all 8 motifs of the tutorial recording you can train a network to help with annotating the rest of the data.

Training requires the audio data and the to be in a [specific format](technical/data_formats). First, export the audio data and the annotations via `File/Export for DAS` to a new folder (not the one containing the original audio)---let's call the folder `quickstart`:

:::{figure-md} xb_export-fig
<img src="images/xb_quick_bird_export.png" alt="export audio and annotations" width=450>

Export audio data and annotations for the whole recording.
:::

Then make a dataset, via _DAS/Make dataset for training_. In the file dialog, select the `quickstart` folder you exported your annotations into. In the next dialog, we will adjust how data is split into training, validation and testing data. For the small data set annotated in the first step of this tutorial, we will not test the model, to maximize the data available for optimizing the network (training and validation). Set the training split to 0.60,  the validation split to 0.40 and the test split to 0.0 (not test):

:::{figure-md} xb_assemble-fig
<img src="images/xb_quick_bird_make_ds.png" alt="assemble dataset" width=600>

Make a dataset for training.
:::

This will create a dataset folder called `quickstart.npy` that contains the audio data and the annotations formatted for training.

## Fast training
Configure a network and start training via _DAS/Train_. This will ask you to select folder with the dataset you just created, `quickstart.npy`. Then, a dialog allows you to configure the network. For the fast training, change the following:
- Set `Chunk duration (samples)` to 4096.
- Set `Number of filters` to 64.
- Set `Filter duration (samples)` to 32.
:::{figure-md} xb_train-fig
<img src="images/xb_quick_bird_train.png" alt="train" width=500>

Train options
:::

Then hit `Start training in GUI`---this will start training in a background process. A small window will display training progress (see also the output in the terminal). Training with this small dataset will finish in ?? minutes on a CPU and in 10 minutes on a GPU. For larger datasets, we highly recommend training on a machine with a discrete Nvidia GPU.

## Predict
Once training finished, we'll generate annotations for a new recording. Load the recording `birdname_130519_110831.1.wav` and  generate predictions using the trained network via _DAS/Predict_. This will ask you to select a model file containing the trained model. Training creates files in the `quickstart.res` folder, starting with the time stamp of training---select the file ending in `_model.h5`.

In the next dialog, predict song for 60 seconds starting after your manual annotations:
- Make sure that `Proof reading mode` is enabled. That way, annotations created by the network will be assigned names ending in `_proposals` - in our case `sine_proposals` and `pulse_proposals`. The proposals will be transformed into proper `sine` and `pulse` annotations during proof reading.
- Enable `Fill gaps shorter than (seconds)` by unchecking the checkbox and set it to 0.005 seconds.
- Enable `Delete segments shorter than (seconds)` by unchecking the checkbox and leave it at the default value of 0.020 seconds.

:::{figure-md} xb_predict-fig
<img src="images/xb_quick_bird_predict.png" alt="predict" width=750>

Predict annotations for a new recording.
:::

In contrast to training, prediction is very fast, and does not require a GPU---it should finish after 10 seconds. The proposed annotations should be already good. Most syllables should be correctly detected but will be some false positive detections, missed syllables, confused syllables, and syllables with imprecise boundaries. We'll fix these predictions errors in the next step.

## Proof reading
To turn the proposals into proper annotations, fix and approve them. Correct any prediction errors---add missing syllables, remove false positive syllables, correct syllable type errors, adjust the syllable timing. See [Create annotations](#create-annotations-manually) and [Edit annotations](#edit-annotations). Once you have corrected all errors in the view, the all proposals in view with `H`.

## Go back to "Export"
Once all proposals have been approved, export the annotations for this file into the same `quickstart` folder, make a new dataset, train, predict, and repeat. If prediction performance is adequate, fully train the network, this time using a completely new recording as the test set and with a larger number of epochs.