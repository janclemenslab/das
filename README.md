# Deep Audio Segmenter (DAS)

_DAS_ automatically annotates animal vocalizations in raw audio recordings using a deep neural network. It can be used through a graphical user interface, from the terminal, or from Python scripts.

## Installation

```shell
conda create -n das -c conda-forge python=3.14 ffmpeg uv -y
conda activate das
uv pip install das --torch-backend=auto
das version
```

See the [installation guide](https://janclemenslab.org/das/installation.html) for CPU- and GPU-specific options.

Users who need the TensorFlow backend should follow the [TensorFlow installation instructions](https://janclemenslab.org/das/install_tf.html) for the final TensorFlow-backed release (0.32.13).

## Documentation

See the [DAS documentation](https://janclemenslab.org/das/) for the complete user guide:

- The quick-start tutorials for [flies](https://janclemenslab.org/das/quickstart_fly.html) and [birds](https://janclemenslab.org/das/quickstart_bird.html) cover manual annotation, network training, and generating new annotations.
- Use the [graphical user interface](https://janclemenslab.org/das/tutorials_gui/tutorials_gui.html).
- Use DAS [from the terminal or Python scripts](https://janclemenslab.org/das/tutorials/tutorials.html).

If you have questions, feedback, or find a bug, please [open an issue](https://github.com/janclemenslab/das/issues).

## Citation

Please cite _DAS_ as:

Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021). _Fast and accurate annotation of acoustic signals with deep neural networks._ [eLife](https://doi.org/10.7554/eLife.68837)

## Acknowledgements

The following packages were modified and integrated into DAS:

- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `das.models.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `das.models.kapre`)

See the corresponding source directories for the original READMEs.
