# Unsupervised classification
_DeepSS-unsupervised_ provides tools for pre-processing acoustic signals for unsupervised classification:

- extract waveforms or spectrograms of acoustic events from a recording
- normalize the duration, center frequency, amplitude, or sign of waveform/spectrograms

Unsupervised classification itself is performed using existing libraries:

- dimensionality reduction: [umap](https://umap-learn.readthedocs.io/)
- clustering: [hdbscan](https://hdbscan.readthedocs.io/) or [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)

Can be Used in combination with [DeepSS](https://github.com/janclemenslab/deepsongsegmenter), a deep learning based method for the supervised annotation of acoustic signals.


## Installation

```shell
pip install deepss-unsupervised
```

## Demos
 Illustration of the workflow and the method using vocalizations from:

- [flies](unsupervised_flies.ipynb)
- [mice](unsupervised_mice.ipynb)
- [birds](unsupervised_birds.ipynb)

![](unsupervised_banner.png)


## Acknowledgements
Code from the following open source packages was modified and integrated into dss-unsupervised:

- [avgn](https://github.com/timsainb/avgn_paper) (Sainburg et al. 2020)
- [noisereduce](https://pypi.org/project/noisereduce)
- [fly pulse classifier](https://github.com/murthylab/MurthyLab_FlySongSegmenter) (Clemens et al. 2018)

Data sources:

- flies: [David Stern](https://www.janelia.org/lab/stern-lab/tools-reagents-data) (Stern, 2014)
- mice: data provided by Kurt Hammerschmidt (Ivanenko et al. 2020)
- birds: [Bengalese finch song repository](https://doi.org/10.6084/m9.figshare.4805749.v5) (Nicholson et al. 2017)


## References

1. T Sainburg, M Thielk, TQ Gentner (2020) Latent space visualization, characterization, and generation of diverse vocal communication signals. Biorxiv . [https://doi.org/10.1101/870311]()

2. J Clemens, P Coen, F Roemschied, T Perreira, D Mazumder, D Aldorando, D Pacheco, M Murthy (2018) Discovery of a New Song Mode in Drosophila Reveals Hidden Structure in the Sensory and Neural Drivers of Behavior. Current Biology 28, 2400–2412.e6 (2018). [https://doi.org/10.1016/j.cub.2018.06.011]()

3. D Stern (2014). Reported Drosophila courtship song rhythms are artifacts of data analysis. BMC Biology

4. A Ivanenko, P Watkins, MAJ van Gerven, K Hammerschmidt, B Englitz (2020) Classifying sex and strain from mouse ultrasonic vocalizations using deep learning. PLoS Comput Biol 16(6): e1007918. [https://doi.org/10.1371/journal.pcbi.1007918]()

5. D Nicholson, JE Queen, S Sober (2017). Bengalese finch song repository. [https://doi.org/10.6084/m9.figshare.4805749.v5]()