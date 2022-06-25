"""Utils for loading and manipulating data for training and prediction."""
import numpy as np
import tensorflow.keras as keras
import dask.array
from typing import Optional, Callable, Sequence, List
from tqdm.autonotebook import tqdm


def unpack_batches(x: np.ndarray, padding: int = 0):
    """[summary]

    Args:
        x ([type]): [description]
        padding (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if padding > 0:
        x = x[:, padding:-padding, ...]
    # reshape everything from [batch, hist, classes] to [time, classes] <- ???
    x = x.reshape((-1, x.shape[-1]))
    return x


def get_data_from_gen(data_gen):
    # PREPEND data_padding samples of zeros/nans so we match len(x)? and postpend nans to match len(x) as well?
    x, y = data_gen.unroll(return_x=True, merge_batches=True)
    # reshape from [batches, nb_hist, ...] to [time, ...]
    x = unpack_batches(x, data_gen.data_padding)
    if y is not None:
        y = unpack_batches(y, data_gen.data_padding)
    return x, y


def sub_range(data_len, fraction: float, min_nb_samples: int = 0, seed=None):
    """[summary]

    Args:
        data_len (int): total length of data
        fraction (float): fraction of data_len to use
        seed (float): seed random number generator for reproducible subset selection

    Returns:
        first_sample (int), last_sample (int)
    """
    np.random.seed(seed)
    sub_len = int(max(np.ceil(fraction * data_len), np.ceil(min_nb_samples)))
    first_sample = np.random.randint(low=0, high=data_len - sub_len - 1)
    last_sample = first_sample + sub_len + 1
    return first_sample, last_sample


def compute_class_weights(y: np.ndarray) -> List[float]:
    """_summary_

    Args:
        y (np.ndarray): [T, nb_classes]

    Returns:
        np.ndarray: nb_classes
    """
    nb_classes = y.shape[1]

    # chunk y
    yy = dask.array.from_array(y)
    nb_chunks = len(yy.chunks[0])

    # count classes over blocks (dask.array.map_blocks does nto work fsr)
    counts = np.zeros((nb_chunks, nb_classes))
    for cnt, block in enumerate(tqdm(yy.blocks, total=nb_chunks, desc='Counting class occurrences')):
        counts[cnt, :] = np.sum(block.compute().astype(float), axis=0)

    # aggregate and normalize
    class_weights = np.sum(counts, axis=0)
    class_weights /= np.sum(class_weights)
    class_weights = [1 / class_weight for class_weight in class_weights]
    return class_weights


class AudioSequence(keras.utils.Sequence):
    """[summary]

    Methods:
        ...
    """

    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None, batch_size: int = 32, shuffle: bool = True, nb_hist: int = 1, y_offset: Optional[int] = None,
                 stride: int = 1, cut_trailing_dim: bool = False, with_y_hist: bool = False, data_padding: int = 0,
                 first_sample: int = 0, last_sample: Optional[int] = None, output_stride: int = 1, nb_repeats: int = 1,
                 shuffle_subset: Optional[float] = None, unpack_channels: bool = False, mask_input: Optional[int] = None,
                 batch_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 class_weights: Optional[Sequence[float]] = None,
                 **kwargs):
        """[summary]

        x and y can be mem-mapped numpy arrays or lazily loaded hdf5 (zarr, xarray) datasets. Dask arrays do not work since they are immutable.
        Args:
            x (np.ndarray): [nb_samples, ...]
            y (np.ndarray, optional): [nb_samples,  nb_classes] - class probabilities - so sum over classes for each sample should be 1.0. Defaults to None.
                                      If None, getitem will only return x batches - neither y nor sample weights
            batch_size (int, optional): number of batches to return. Defaults to 32.
            shuffle (bool, optional): randomize order of batches. Defaults to True.
            nb_hist (int, optional): nb of time steps per batch. Defaults to 1.
            y_offset (int, optional): time offset between x and y. nb_hist/2 if None (predict central sample in each batch). Defaults to None.
            stride (int, optional): nb of time steps between batches. Defaults to 1.
            cut_trailing_dim (bool, optional): Remove trailing dimension. Defaults to False.
            with_y_hist (bool, optional): y as central value of the x_hist window (False) or the full sequence covering
                                          the x_hist window (True). Defaults to False.
            data_padding (int, optional): if > 0, will set weight of as many samples at start and end of nb_hist window to zero. Defaults to 0.
            first_sample (int): 0
            last_sample (int, optional): None - last_sample in x, otherwise last_sample
            output_stride (int): Take every Nth sample as output. Useful in combination with a "downsampling frontend". Defaults to 1 (every sample).
            nb_repeats (int): Number of repeats before the dataset runs out of data. Defaults to 1 (no repeats).
            shuffle_subset (float): Fraction of batches to use - only works if shuffle=True
            unpack_channels (bool): For multi-channel models with single-channel preprocessing -
                                    unpack [nb_hist, nb_channels] -> [nb_channels * [nb_hist, 1]]
            mask_input (int): half width of the number of central samples to mask. Defaults to None (no masking).
            batch_processor (Callable[[np.ndarray], np.ndarray], optional): For augmentations. Defaults to None.
            class_weights (Sequence[float], optional): Weights for each class used for balancing. Defaults to None (no balancing).
        """
        # TODO clarify "channels" semantics
        self.x, self.y = x, y

        self.first_sample = first_sample
        self.last_sample = self.x.shape[0] if last_sample is None else last_sample
        self.nb_samples = self.last_sample - self.first_sample
        self.nb_repeats = nb_repeats
        self.output_stride = output_stride
        self.with_y = False if self.y is None else True
        if self.with_y:
            self.nb_classes = self.y.shape[-1]
        else:
            self.nb_classes = 0  # if no y data

        self.batch_size = batch_size
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_subset = shuffle_subset
        self.x_hist = nb_hist
        self.with_y_hist = with_y_hist
        self.data_padding = data_padding
        self.unpack_channels = unpack_channels
        self.class_weights = class_weights
        self.mask_input = mask_input
        s0 = self.first_sample / self.stride
        s1 = (self.last_sample - self.x_hist - 1) / self.stride
        self.allowed_batches = np.arange(s0, s1, dtype=np.uintp)  # choose from all existing batches
        if self.shuffle_subset is not None:  # only choose from a fixed subset of existing batches
            self.allowed_batches = np.random.choice(self.allowed_batches,
                                                    size=int(len(self.allowed_batches) * self.shuffle_subset),
                                                    replace=False)

        if y_offset is None:
            self.y_offset = int(self.x_hist / 2)
        else:
            self.y_offset = int(y_offset)

        # ignore padding samples at beginning and end of x_hist to avoid boundary conditions problems
        if self.with_y_hist:
            self.weights = np.ones((self.batch_size, self.x_hist))
            if self.data_padding > 0:
                self.weights[:, :self.data_padding] = 0
                self.weights[:, -self.data_padding:] = 0
            self.weights = self.weights[:, ::output_stride]
        else:
            self.weights = np.ones((self.batch_size,))

        self.batch_processor = batch_processor
        self._idx_offset = 0

    def unroll(self, return_x=True, merge_batches=True):
        """[summary]

        Args:
            return_x=True
            merge_batches=True

        Returns:
            [type]: [description]
            (xx, yy), (None, yy) or (xx,)
        """
        xx = None
        if return_x:
            xx = np.zeros((len(self), self.batch_size, self.x_hist, *self.x.shape[1:]))  # this is probably more general
        if self.with_y_hist:
            yy = np.zeros((len(self), self.batch_size, int(self.x_hist / self.output_stride), self.nb_classes)) if self.with_y else None
        else:
            yy = np.zeros((len(self), self.batch_size, self.nb_classes)) if self.with_y else None

        for cnt, gen_output in enumerate(self):
            if return_x:
                if self.unpack_channels:
                    xx[cnt, ...] = np.concatenate(gen_output[0], axis=-1)
                else:
                    xx[cnt, ...] = gen_output[0]
            if self.with_y:
                yy[cnt, ...] = gen_output[1]

        if merge_batches:
            if return_x:
                xx = xx.reshape((len(self) * self.batch_size, self.x_hist, *self.x.shape[1:]))  # this is probably more general
            if self.with_y_hist:
                yy = yy.reshape((len(self) * self.batch_size, int(self.x_hist / self.output_stride), self.nb_classes)) if self.with_y else None
            else:
                yy = yy.reshape((len(self) * self.batch_size, self.nb_classes)) if self.with_y else None

        if self.with_y:
            out = (xx, yy)
        else:
            out = (xx, )
        return out

    def __len__(self):
        """Number of batches."""
        return int(self.nb_repeats * max(0, np.floor((self.nb_samples - ((self.stride*(self.batch_size-1)) + self.x_hist)) / (self.stride * (self.batch_size))) + 1))

    def __str__(self):
        string = ['AudioSequence with {} batches each with {} items.\n'.format(len(self), self.batch_size),
                  '   Total of {} samples with\n'.format(self.nb_samples),
                  '   each x={} and\n'.format(self.x.shape[1:])]
        string.append('   each y={}'.format(self.y.shape[1:])) if self.y is not None else 'no y.'
        return ''.join(string)

    def __getitem__(self, idx):
        """Get item from AudioSequence

        Args:
            idx (int): batch

        Returns:
            batch_x [np.ndarray]: [nb_batches, nb_hist, ..., nb_classes].
            batch_y [np.ndarray]: [nb_batches, nb_hist, nb_classes].
            weights_y [np.ndarray]: [nb_batches, nb_hist, nb_classes].
        """
        idx += self._idx_offset
        batch_x = np.zeros((self.batch_size, self.x_hist, *self.x.shape[1:]), dtype=self.x.dtype)

        if self.with_y:
            if self.with_y_hist:
                batch_y = np.zeros((self.batch_size, int(self.x_hist / self.output_stride), self.nb_classes), dtype=self.y.dtype)
            else:
                batch_y = np.zeros((self.batch_size, self.nb_classes), dtype=self.y.dtype)

        if self.shuffle:
            # pts = np.random.randint(self.first_sample / self.stride, (self.last_sample - self.x_hist - 1) / self.stride, self.batch_size)
            pts = np.random.choice(self.allowed_batches,
                                   size=self.batch_size,
                                   replace=False)
        else:
            pts = range(int(self.first_sample/self.stride) + idx * self.batch_size,
                        int(self.first_sample/self.stride) + (idx + 1) * self.batch_size)

        for cnt, bat in enumerate(pts):
            batch_x[cnt, ...] = self.x[int(bat * self.stride):int(bat * self.stride + self.x_hist), ...].copy()

            if self.with_y:
                if self.with_y_hist:
                    batch_y[cnt, ...] = self.y[int(bat * self.stride):int(bat * self.stride + self.x_hist):self.output_stride, ...]
                else:
                    batch_y[cnt, ...] = self.y[int(bat * self.stride + self.y_offset), ...]
        if self.unpack_channels:
            batch_x = [batch_x[..., chn][..., np.newaxis] for chn in range(batch_x.shape[-1])]

        # "mask" input
        if self.mask_input is not None:
            batch_x[:, int(batch_x.shape[1]/2 - self.mask_input):int(batch_x.shape[1]/2 + self.mask_input), :] = 0

        if self.batch_processor is not None:
            batch_x = self.batch_processor(batch_x)

        if self.with_y:
            out = (batch_x, batch_y)

            if self.class_weights is not None:
                # weights contain weight for the class at each time point
                weights = np.zeros_like(self.weights)
                labels = np.argmax(batch_y, axis=-1)
                for label, weight in enumerate(self.class_weights):
                    weights[labels == label] = weight
                weights *= self.weights  # use weights as mask to zero boundaries
            else:
                weights = self.weights

            if self.data_padding > 0:
                out = (batch_x, batch_y, weights)
        else:
            out = (batch_x, )
        return out
