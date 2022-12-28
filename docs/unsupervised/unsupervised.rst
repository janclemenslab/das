
.. _uns:


Unsupervised classification
===========================

Unsupervised classifications is an alternative to tedious manual
classification of song types: Use *DAS* via the GUI or the command line
to detect anything that you think is song and then classify song into
different types afterwards. The song types discovered with unsupervised
methods can then be used to create a training dataset for training *DAS*
to directly label the different song types.

*DAS-unsupervised* provides tools for applying this approach with a
focus on pre-processing acoustic signals for unsupervised
classification:

-  extract waveforms or spectrograms of acoustic events from a recording
-  normalize the duration, center frequency, amplitude, or sign of
   waveform/spectrograms

Unsupervised classification itself is performed using existing
libraries:

-  dimensionality reduction:
   `umap <https://umap-learn.readthedocs.io/>`__
-  clustering: `hdbscan <https://hdbscan.readthedocs.io/>`__ or
   `scikit-learn <https://scikit-learn.org/stable/modules/clustering.html>`__

Code for the unsupervised classification can be found at
https://github.com/janclemenslab/DAS_unsupervised.

Examples
--------

We illustrate different pre-processing and classification strategies
using three different examples

-  `Courtship song of flies <flies.html>`__ - normalize waveforms to
   automatically detect sine song and different pulse types.
-  `Song of Bengalese finches <birds.html>`__ - process syllable
   spectrograms to classify >40 syllable types.
-  `Ultrasonic vocalizations from mice <mice.html>`__ - process
   syllable spectrograms to group syllables by the shape of their
   spectral contours.

|image1|

Acknowledgements
----------------

Code from the following open source packages was modified and integrated
into das-unsupervised:

-  `avgn <https://github.com/timsainb/avgn_paper>`__ (Sainburg et
   al. 2020)
-  `noisereduce <https://pypi.org/project/noisereduce>`__
-  `fly pulse
   classifier <https://github.com/murthylab/MurthyLab_FlySongSegmenter>`__
   (Clemens et al. 2018)

Data sources:

-  flies: `David
   Stern <https://www.janelia.org/lab/stern-lab/tools-reagents-data>`__
   (Stern, 2014)
-  birds: `Bengalese finch song
   repository <https://doi.org/10.6084/m9.figshare.4805749.v5>`__
   (Nicholson et al. 2017)
-  mice: `Data from Ivanenko et al.
   (2020) <https://public.data.donders.ru.nl/dcn/DSC_620840_0003_891_v1>`__


References
----------

1. T Sainburg, M Thielk, TQ Gentner (2020) Latent space visualization,
   characterization, and generation of diverse vocal communication
   signals. Biorxiv . https://doi.org/10.1101/870311

2. J Clemens, P Coen, F Roemschied, T Perreira, D Mazumder, D Aldorando,
   D Pacheco, M Murthy (2018) Discovery of a New Song Mode in Drosophila
   Reveals Hidden Structure in the Sensory and Neural Drivers of
   Behavior. Current Biology 28, 2400–2412.e6 (2018).
   https://doi.org/10.1016/j.cub.2018.06.011

3. D Stern (2014). Reported Drosophila courtship song rhythms are
   artifacts of data analysis. BMC Biology

4. A Ivanenko, P Watkins, MAJ van Gerven, K Hammerschmidt, B Englitz
   (2020) Classifying sex and strain from mouse ultrasonic vocalizations
   using deep learning. PLoS Comput Biol 16(6): e1007918.
   https://doi.org/10.1371/journal.pcbi.1007918

5. D Nicholson, JE Queen, S Sober (2017). Bengalese finch song
   repository. https://doi.org/10.6084/m9.figshare.4805749.v5

.. |image1| image:: banner.png


.. toctree::
   :maxdepth: 1
   :hidden:

   :toctree: uns
   flies
   birds
   mice
