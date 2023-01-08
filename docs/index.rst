
Welcome to *DAS*
================

*DAS* — short for *Deep Audio Segmenter* — is a tool for annotating song
in audio recordings. At the core of *DAS* is a deep neural network,
implemented in Tensorflow. The network takes single- and multi-channel
audio as an input and returns the probability of finding a particular
song type for each audio sample. *DAS* can be used with a graphical user
interface for loading audio data, annotating song manually, training a
network, and generating annotations on audio. Alternatively, *DAS* can
be used programmatically from the command line, in python notebooks, or
in your own python code via the ``das`` module.

If you use *DAS*, please cite:

Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan
Clemens (2021). *Fast and accurate annotation of acoustic signals with
deep neural networks*, eLife,
`https://doi.org/10.7554/eLife.68837 <https://doi.org/10.7554/eLife.68837>`__

.. panels ::

    .. link-button :: installation
        :text: Installation
        :type: ref
        :classes: stretched-link


Tutorials
---------

.. panels ::

    .. link-button :: quickstart_fly
        :text: Quick start tutorial (fly)
        :type: ref
        :classes: stretched-link
    +++

    Annotate song, train a network, and predict on new samples.

    ---

    .. link-button :: tutorials_gui/tutorials_gui
        :text: Using the GUI
        :type: ref
        :classes: stretched-link

    +++

    Comprehensive description of all GUI dialogs and options.

    ---

    .. link-button :: tutorials/tutorials
        :text: Use in python and from the terminal
        :type: ref
        :classes: stretched-link

    +++

    Convert your own data, train and evaluate a network, predict on new samples in realtime.

    ---

    .. link-button :: unsupervised/unsupervised
        :text: Classify
        :type: ref
        :classes: stretched-link

    +++

    Discover song types in annotated syllables.


Technical documentation
-----------------------

.. panels ::

    .. link-button :: technical/technical
        :text: Technical details
        :type: ref
        :classes: stretched-link
    +++

    Command-line interface and data structures.

    ---

    .. link-button :: api
        :text: Developer API
        :type: ref
        :classes: stretched-link

    +++

    Comprehensive description of classes and functions.




.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   Introduction <self>
   installation
   quickstart_fly
   tutorials_gui/tutorials_gui
   tutorials/tutorials
   unsupervised/unsupervised
   technical/technical
   Developer API <api>
