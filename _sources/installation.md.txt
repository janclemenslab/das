# Installation

## Pre-requisites

_Anaconda python_: Install the [anaconda python distribution](https://docs.anaconda.com/anaconda/install/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html)). If conda is already installed on your system, make sure you have conda v4.8.4+. If not, update from an older version with `conda update conda`.

_Libsoundfile (Linux only)_: If you are on Linux and want to load audio from a wide range of audio formats (other than `wav`), then you need to install `libsndfile`. The GUI uses the [soundfile](http://pysoundfile.readthedocs.io/) python package, which relies on `libsndfile`. `libsndfile` will be automatically installed on Windows and macOS. On Linux, the library needs to be installed manually with: `sudo apt-get install libsndfile1`. Again, this is only required if you work with more exotic audio files.

_Visual C++ runtime (Windows only)_: This is typically installed so only required if _DAS_ fails to load the native tensorflow runtime. Download the latest version [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

## Install _DAS_
Create an anaconda environment called `das` that contains all the required packages.

On windows
```shell
conda install mamba -c conda-forge -n base -y
mamba create python=3.9 das=0.32.3 "numpy<1.24" -c conda-forge -c ncb -c anaconda -c nvidia -n das -y
```

On Linux or MacOS (intel and arm):
```shell
conda install mamba -c conda-forge -n base -y
mamba create python=3.10 das=0.32.3 -c conda-forge -c ncb -c anaconda -c nvidia -c apple -n das -y
```


## Test the installation (Optional)
To quickly test the installation, run these  commands in the terminal:
```shell
conda activate das  # activate the conda environment
das train --help  # test DAS training
das gui  # start the DAS GUI
```
The second command will display the command line arguments for `das train`. The last command, `das gui`, will start the GUI. This step will *not* work with the no-GUI install.

## Make a desktop icon (Optional)
To start the _DAS_ GUI without having to use a terminal, create a clickable startup script on the desktop.

On macOS or Linux, place a text file called `das.sh` (linux) or `das.command` (macOS) with the following content on your desktop:
```shell
# /bin/bash
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate das
das gui
```
Make the files executable with `chmod +x FILENAME`, where FILENAME is `das.sh` (linux) or `das.command` (macOS).

For windows, place a text file called `das.bat` with the following content on the desktop:
```shell
TITLE DAS
CALL conda.bat activate das
das gui
```

## Next steps
If all is working, you can now use _DAS_ to annotate song. To get started, you will first need to train a network on your own data. For that you need annotated audio - either create new annotations [using the GUI](/tutorials_gui/tutorials_gui) or convert existing annotations [using python scripts](/tutorials/tutorials).
