"""Plot utilities."""
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib_scalebar
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def scalebar(length, dx=1, units='', label=None, axis=None, location='lower right', frameon=False, **kwargs):
    """Add scalebar to axis.

    Usage:
        plt.subplot(122)
        plt.plot([0,1,2,3], [1,4,2,5])
        scalebar(0.5, 'femtoseconds', label='duration', location='lower right’)

    Args:
        length (float): Length of the scalebar in units of axis ticks - length of 1.0 corresponds to spacing between to major x-ticks
        dx (int, optional): Scale factor for length. E.g. if scale factor is 10, the scalebar of length 1.0 will span 10 ticks. Defaults to 1.
        units (str, optional): Unit label (e.g. 'milliseconds'). Defaults to ''.
        label (str, optional): Title for scale bar (e.g. 'Duration'). Defaults to None.
        axis (matplotlib.axes.Axes, optional): Axes to add scalebar to. Defaults to None (currently active axis - plt.gca()).
        location (str, optional): Where in the axes to put the scalebar (upper/lower/'', left/right/center). Defaults to 'lower right'.
        frameon (bool, optional): Add background (True) or not (False). Defaults to False.
        kwargs: location=None, pad=None, border_pad=None, sep=None,
                frameon=None, color=None, box_color=None, box_alpha=None,
                scale_loc=None, label_loc=None, font_properties=None,
                label_formatter=None, animated=False):

    Returns:
        Handle to scalebar object
    """

    if axis is None:
        axis = plt.gca()

    if 'dimension' not in kwargs:
        kwargs['dimension'] = matplotlib_scalebar.dimension._Dimension(units)

    scalebar = ScaleBar(dx=dx, units=units, label=label, fixed_value=length, location=location, frameon=frameon, **kwargs)
    axis.add_artist(scalebar)
    return scalebar


def remove_axes(axis=None, all=False):
    """Remove top & left border around plot or all axes & ticks.

    Args:
        axis (matplotlib.axes.Axes, optional): Axes to modify. Defaults to None (currently active axis - plt.gca()).
        all (bool, optional): Remove all axes & ticks (True) or top & left border only (False). Defaults to False.
    """
    if axis is None:
        axis = plt.gca()

    # Hide the right and top spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')

    if all:
        # Hide the left and bottom spines
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        # Remove all tick labels
        axis.yaxis.set_ticks([])
        axis.xaxis.set_ticks([])



import string
from itertools import cycle

def label_axes(fig=None, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if fig is None:
        fig = plt.gcf()

    if labels is None:
        labels = string.ascii_uppercase

    if 'size' not in kwargs:
        kwargs['size'] = 13

    if 'weight' not in kwargs:
        kwargs['weight'] = 'bold'
    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (-0.2, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)

def downsample_plot(x,y, ds=20):
    """Reduces complexity of exported pdfs w/o impairing visual appearance.

    Modified from pyqtgraph - downsampleMethod=peak.
    Keeps peaks so the envelope of the waveform is preserved.
    """
    n = len(x) // ds
    x1 = np.empty((n,2))
    x1[:] = x[:n*ds:ds,np.newaxis]
    x0 = x1.reshape(n*2)
    y1 = np.empty((n,2))
    y2 = y[:n*ds].reshape((n, ds))
    y1[:,0] = y2.max(axis=1)
    y1[:,1] = y2.min(axis=1)
    y0 = y1.reshape(n*2)
    return x0, y0

class Pdf:
    """Thin wrapper around Autosaving variant"""

    def __init__(self, savename, autosave=True, **savefig_kws):
        self.pdf = PdfPages(savename)
        self.autosave = autosave
        self.savefig_kws = savefig_kws

    def __enter__(self):
        self.pdf.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.autosave:
            self.pdf.savefig(**self.savefig_kws)
        self.pdf.__exit__(exc_type, exc_val, exc_tb)
