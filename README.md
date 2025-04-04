<!-- [![Test install](https://github.com/janclemenslab/das/actions/workflows/main.yml/badge.svg)](https://github.com/janclemenslab/das/actions/workflows/main.yml) -->

# Deep Audio Segmenter
_DAS_ is a method for automatically annotating song from raw audio recordings based on a deep neural network. _DAS_ can be used with a graphical user interface, from the terminal, or from within python scripts.

If you have questions, feedback, or find bugs please raise an [issue](https://github.com/janclemenslab/das/issues).

Please cite _DAS_ as:

Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021).
_Fast and accurate annotation of acoustic signals with deep neural networks._
[eLife](https://doi.org/10.7554/eLife.68837)


See the documentation at [https://janclemenslab.org/das/](https://janclemenslab.org/das/) for instructions on how to [install DAS](https://janclemenslab.org/das/installation.html) and for a user guide:

- A [quick start tutorial](https://janclemenslab.org/das/quickstart.html) walks through all steps from manually annotating song, over training a network, to generating new annotations.
- How to use the [graphical user interface](https://janclemenslab.org/das/tutorials_gui/tutorials_gui.html).
- How to use _DAS_ [from the terminal or from python scripts](https://janclemenslab.org/das/tutorials/tutorials.html).



## Acknowledgements
The following packages were modified and integrated into das:

- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `das.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `das.kapre`)

See the sub-module directories for the original READMEs.
