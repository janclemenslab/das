"""Plot utilities."""
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib_scalebar
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import itertools


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


def remove_axes(axis=None, all=False, which='tblr'):
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


def despine(which='tr', axis=None):

    sides = {'t': 'top', 'b': 'bottom', 'l': 'left', 'r': 'right'}

    if axis is None:
        axis = plt.gca()

    # Hide the spines
    for side in which:
        axis.spines[sides[side]].set_visible(False)

    # Hide the tick marks and labels
    if 'r' in which:
        axis.yaxis.set_ticks_position('left')

    if 't' in which:
        axis.xaxis.set_ticks_position('bottom')

    if 'l' in which:
        axis.yaxis.set_ticks([])

    if 'b' in which:
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

    # if 'weight' not in kwargs:
    #     kwargs['weight'] = 'bold'
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

    def __init__(self, savename, autosave=True, style=None, **savefig_kws):
        """[summary]

        Args:
            savename ([type]): [description]
            autosave (bool, optional): [description]. Defaults to True.
            style ([type], optional): [description]. Defaults to None.
        """
        self.pdf = PdfPages(savename)
        self.autosave = autosave
        self.style = style
        self.orig = None
        self.savefig_kws = savefig_kws

    def __enter__(self):
        if self.style is not None:
            self.orig = mpl.rcParams.copy()
            mpl.style.use(self.style)
        self.pdf.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.autosave:
            self.pdf.savefig(**self.savefig_kws)
        if self.orig is not None:
            dict.update(mpl.rcParams, self.orig)
        self.pdf.__exit__(exc_type, exc_val, exc_tb)


# from https://gist.github.com/thesamovar/52dbbb3a58a73c590d54c34f5f719bac
def panel_specs(layout, fig=None):
    """
    ```
    layout = '''
    AAB
    AA.
    .DD
    '''
    fig = plt.figure(figsize=(10, 7))
    axes, spec = panels(layout, fig=fig)
    spec.set_width_ratios([1, 3, 1])
    label_panels(axes, letters='ABD')
    plt.tight_layout()
    ```

    Args:
        layout ([type]): [description]
        fig ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # default arguments
    if fig is None:
        fig = plt.gcf()
    # format and sanity check grid
    lines = layout.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    linewidths = set(len(line) for line in lines)
    if len(linewidths)>1:
        raise ValueError('Invalid layout (all lines must have same width)')
    width = linewidths.pop()
    height = len(lines)
    panel_letters = set(c for line in lines for c in line)-set('.')
    # find bounding boxes for each panel
    panel_grid = {}
    for letter in panel_letters:
        left = min(x for x in range(width) for y in range(height) if lines[y][x]==letter)
        right = 1+max(x for x in range(width) for y in range(height) if lines[y][x]==letter)
        top = min(y for x in range(width) for y in range(height) if lines[y][x]==letter)
        bottom = 1+max(y for x in range(width) for y in range(height) if lines[y][x]==letter)
        panel_grid[letter] = (left, right, top, bottom)
        # check that this layout is consistent, i.e. all squares are filled
        valid = all(lines[y][x]==letter for x in range(left, right) for y in range(top, bottom))
        if not valid:
            raise ValueError('Invalid layout (not all square)')
    # build axis specs
    gs = mpl.gridspec.GridSpec(ncols=width, nrows=height, figure=fig)
    specs = {}
    for letter, (left, right, top, bottom) in panel_grid.items():
        specs[letter] = gs[top:bottom, left:right]
    return specs, gs

def panels(layout, fig=None):
    # default arguments
    if fig is None:
        fig = plt.gcf()
    specs, gs = panel_specs(layout, fig=fig)
    for letter, spec in specs.items():
        axes[letter] = fig.add_subplot(spec)
    return axes, gs

def label_panel(ax, letter, *, prefix='', postfix='.', spaces=6, pad=10, fontsize=18):
    ax.set_title(prefix+letter+postfix+' '*spaces, loc='left', pad=pad,
                 fontdict={'horizontalalignment': 'right',
                           'fontsize': fontsize})

def label_panels(axes, letters=None, *, prefix='', postfix='.', spaces=6, pad=10, fontsize=18):
    if letters is None:
        letters = axes.keys()
    for letter in letters:
        ax = axes[letter]
        label_panel(ax, letter, prefix=prefix, postfix=postfix, spaces=spaces, pad=pad, fontsize=fontsize)



import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
# from https://stackoverflow.com/a/60345118/2301098
def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists,
    but are used as row and column headers, looking like this:

    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |


    Parameters
    ----------

    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    title_label : str, optional
        Label for the top left corner in the legend table.

    ncol : int
        Number of columns.


    Other Parameters
    ----------------

    Refer to `matplotlib.legend.Legend` for other parameters.

    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]

        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

        # empty label
        empty = [""]

        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow

        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


def imshow_text(data, labels=None, ax=None, color_high='w', color_low='k', color_threshold=50, skip_zeros=False):
    """Text labels for individual cells of an imshow plot

    Args:
        data ([type]): Color values
        labels ([type], optional): Text labels. Defaults to None.
        ax ([type], optional): axis. Defaults to plt.gca().
        color_high (str, optional): [description]. Defaults to 'w'.
        color_low (str, optional): [description]. Defaults to 'k'.
        color_threshold (int, optional): [description]. Defaults to 50.
        skip_zeros (bool, optional): [description]. Defaults to False.
    """
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = data

    for x, y in itertools.product(range(data.shape[0]), range(data.shape[1])):
        if labels[y, x] == 0:
            continue
        ax.text(x, y, f'{labels[y, x]:1.0f}',
                ha='center', va='center',
                c=color_high if data[y, x]>color_threshold else color_low)


def bar_text(ax=None, spacing=-20, to_int=True):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate. Defaults to current axis.
        spacing (int): The distance between the labels and the bars. Defaults to -20 (inside bar)
    """
    if ax is None:
        ax = plt.gca()

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if to_int:
            label = int(y_value)
        else:
            label = float(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,
            color='w')                      # Vertically align label differently for
                                        # positive and negative values.
