"""Utils for loading and manipulating data for training and prediction."""
import numpy as np
import keras
import sklearn.model_selection
import copy


def events_to_trace(event_times, event_duration: int = 60, trace_duration: int = None):
    """[summary]
    
    Args:
        event_times ([type]): List of event_times in units of samples - if you have event_times in units of seconds and want a trace with 10_000Hz, then multiply each event_times by that sampling rate.
        event_duration (int, optional): [description]. Defaults to 60 samples.
        trace_duration (int, optional): [description]. Defaults to None - (max(event_times) + 2 * even_duration) samples.
    Returns:
        trace - [event_duration, 2]  p(no event) and p(event) correspond to [...,0] and [...,1], respectively.
    """
    event_times = event_times.astype(np.uintp)  # ensure these are proper indices

    if trace_duration is None:
        trace_duration = np.max(event_times) + 2 * event_duration    
    trace = np.zeros((trace_duration, 2), dtype=np.bool)    

    # now "smooth" the trace so that events have an extent in time, np.clip is taking care of boundary conditions
    for stp in np.arange(-event_duration, event_duration).astype(np.intp):
        trace[np.clip(event_times + stp, 0, trace_duration), 1] = 1    
    trace[:,0] = 1 - trace[:,1]  # 0 index is p(no event)
    return trace


def make_training_data(x, y, save_dir: str, x_test=None, y_test=None, random_split: bool = False, val_fraction: float = 0.05):
    """[summary]
    
    Args:
        x ([type]): [nb_samples, ...] - input features, e.g. [samples, channels] or [samples, wavelet_coefs, chhannels]
        y ([type]): [nb_samples, nb_classes] - class probabilities 
        save_dir (str): [description]
        x_test ([type], optional): [description]. Will save only if both x_test and y_test are provided.
        y_test ([type], optional): [description]. Will save only if both x_test and y_test are provided.
        random_split (bool, optional): [description]. Defaults to False. If True, will split random samples into train/val. If False will split off last val_fraction samples for validation.
        val_fraction (float, optional): [description]. Defaults to 0.05. Fraction of samples to split off for validation
    """

    if random_split:
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, test_size=val_fraction, random_state=42, stratify=np.argmax(y, axis=1))
    else:
        val_range = int(x.shape[0] * val_fraction)
        x_train, y_train = x[:-val_range], y[:-val_range]
        x_val, y_val = x[-val_range:], y[-val_range:]

    np.save(save_dir + '/x_train.npy', x_train)
    np.save(save_dir + '/y_train.npy', y_train)
    np.save(save_dir + '/x_val.npy', x_val)
    np.save(save_dir + '/y_val.npy', y_val)
    if (x_test is not None) and (x_test is not None):
        np.save(save_dir + '/x_test.npy', x_test)
        np.save(save_dir + '/y_test.npy', y_test)
    

def load_data(data_dir='../dat.preprocessed'):
    """Load train/val/test data.

    Args:
        data_dir (str, optional): Defaults to '../dat.preprocessed'.

    Returns:
        x_train, y_train: mem-mapped numpy arrays
        x_val, y_val, x_test, y_test, song_test, pulse_times_test: numpy arrays
    """

    x_test = np.load(data_dir + '/x_test.npy').astype(np.float)
    y_test = np.load(data_dir + '/y_test.npy').astype(np.float)
    try:
        song_test = np.load(data_dir + '/song_test.npy').astype(np.float)
    except FileNotFoundError:
        song_test = None

    try: 
        pulse_times_test = np.load(data_dir + '/pulse_times_test.npy').astype(np.float)
    except FileNotFoundError:
        pulse_times_test = None

    x_val = np.load(data_dir + '/x_val.npy')
    y_val = np.load(data_dir + '/y_val.npy')

    x_train = np.lib.format.open_memmap(data_dir + '/x_train.npy', 'r')
    y_train = np.lib.format.open_memmap(data_dir + '/y_train.npy', 'r')
    return x_train, y_train, x_val, y_val, x_test, y_test, song_test, pulse_times_test


def merge_labels(y, mode: int):
    """Merge label types.

    Args:
        y (np.ndarray): [nb_hist, nix/pulse/sine]
        mode (int): 0 - preserves all three classes, 1 - remove sine, predict only pulse, 2 - remove pulse, predict sine, 3 - predict any song (pulse or sine)

    Returns:
        np.ndarray: modified y
    """
    NIX, PULSE, SINE = 0, 1, 2
    labels_to_merge = {0: None, 1: [NIX, SINE], 2: [NIX, PULSE], 3: [PULSE, SINE]}

    if labels_to_merge[mode] is not None:
        lm = labels_to_merge[mode]
        y[..., lm[0]] = np.sum(y[..., lm], axis=-1)
        y = np.delete(y, lm[-1], axis=-1)  # and remove last of the merged labels
    
    return y


def unpack_batches(x, padding=0):
    """[summary]
    
    Args:
        x ([type]): [description]
        padding (int, optional): [description]. Defaults to 0.
    
    Returns:
        [type]: [description]
    """
    if padding > 0:
        x = x[:, padding:-padding, ...]
    # reshape everything from [batch, hist, classes] to [time, classes]
    x = x.reshape((-1, x.shape[-1]))
    return x


def sub_range(data_list, fraction: float, min_nb_samples: int = 0, seed=None):
    """[summary]
    
    Args:
        data_list ([type]): [description]
        fraction ([type]): [description]
        seed (float): seed random number generator for reproducible subset selection
    
    Returns:
        [type]: [description]
    """
    np.random.seed(seed)
    sub_len = max(int(fraction * data_list[0].shape[0]), int(min_nb_samples))
    sub_start = np.random.randint(low=0, high=data_list[0].shape[0] - sub_len - 1)
    sub_end = sub_start + sub_len + 1
    data_list = [d[sub_start:sub_end] for d in data_list]
    return data_list


class AudioSequence(keras.utils.Sequence):
    """[summary]
    
    Methods:
        for_prediction - get a copy of the sequence that only returns x values - no y or weights - necessary for prediction
        unroll - 
    """

    def __init__(self, x, y=None, batch_size=32, shuffle=True, nb_hist=1, y_offset=None, stride=1, mode=0,
                 cut_trailing_dim=False, compute_power=False, with_y_hist=False, data_padding=0,
                 **kwargs):
        """[summary]

        x and y can be mem-mapped numpy arrays or lazily loaded hdf5 (zarr, xarray) datasets. Dask arrays do not work since they are immutable.
        Args:
            x (np.ndarray): [nb_samples, nb_channels]
            y (np.ndarray, optional): [nb_samples,  nb_classes] - class probabilities - so sum over classes for each sample should be 1.0. Defaults to None. If none, getitem will only return x batches - neither y nor sample weights
            batch_size (int, optional): number of batches to return. Defaults to 32.
            shuffle (bool, optional): randomize order of batches. Defaults to True.
            nb_hist (int, optional): nb of time steps per batch. Defaults to 1.
            y_offset ([type], optional): time offset between x and y. nb_hist/2 if None (predict central sample in each batch). Defaults to None.
            stride (int, optional): nb of time steps between batches. Defaults to 1.
            mode (int, optional): merge classes. if 0 - preserves all three classes, 1 - nix, pulse, 2 - nix, sine, 3 - nix, pulse+sine. Defaults to 0.
            cut_trailing_dim (bool, optional): Remove trailing dimension. Defaults to False.
            computer_power (bool, optional): Merge trailing dimension by computing the power over it. Defaults to False.
            with_y_hist (bool, optional): y as central value of the x_hist window (False) or the full sequence covering the x_hist window (True). Defaults to False.
            data_padding (int, optional): if > 0, will set weight of as many samples at start and end of nb_hist window to zero. Defaults to 0.
        """

        self.x, self.y = x, y
        self.nb_samples = self.x.shape[0]
        self.nb_channels = self.x.shape[1]
        self.with_y = False if self.y is None else True
        if self.with_y:
            self.nb_classes = self.y.shape[-1]
        else:
            self.nb_classes = 0
        self.mode = mode
        if self.mode > 0:
            self.nb_classes -= 1

        self.batch_size = batch_size
        self.stride = stride
        self.shuffle = shuffle
        self.x_hist = nb_hist
        self.with_y_hist = with_y_hist
        self.cut_trailing_dim = cut_trailing_dim
        self.compute_power = compute_power
        self.data_padding = data_padding

        if y_offset is None:
            self.y_offset = int(self.x_hist / 2)
        else:
            self.y_offset = int(y_offset)
        
        # ignore padding samples at beginning and end of x_hist to avoid boundary conditions problems
        if self.data_padding > 0:
            self.weights = np.ones((self.batch_size, self.x_hist))
            self.weights[:, :self.data_padding] = 0
            self.weights[:, -self.data_padding:] = 0
        else:
            self.weights = None

    def for_prediction(self):
        """returns variant of the AudioSequence that yields only x-data."""
        gen = copy.copy(self)
        gen.with_y = False
        return gen

    def unroll(self, merge_batches=True):
        """Unroll the generator.
        
        Args:
            merge_batches (bool, optional): If True, will reshape unrolled data to [time, ...], 
                                            otherwise will be [batches, x_hist, ...]. Defaults to True.
        
        Returns:
            Unrolled generator - either as
        """
        # TODO make this work with classification (vs. prediction) - need to ignore self.x_hist
        xx = np.zeros((len(self), self.batch_size, self.x_hist, self.nb_channels))
        if self.with_y_hist:
            yy = np.zeros((len(self), self.batch_size, self.x_hist, self.nb_classes)) if self.with_y else None
        else:
            yy = np.zeros((len(self), self.batch_size, self.nb_classes)) if self.with_y else None

        for cnt, gen_output in enumerate(self):
            xx[cnt, ...] = gen_output[0]
            if self.with_y:
                yy[cnt, ...] = gen_output[1]

        if merge_batches:
            xx = xx.reshape((len(self) * self.batch_size, self.x_hist, self.nb_channels))
            if self.with_y_hist:
                yy = yy.reshape((len(self) * self.batch_size, self.x_hist, self.nb_classes)) if self.with_y else None
            else:
                yy = yy.reshape((len(self) * self.batch_size, self.nb_classes)) if self.with_y else None

        if self.with_y:
            out = (xx, yy)
        else:
            out = (xx, )
        return out

    def __len__(self):
        """Number of batches."""
        return int(max(0, np.floor((self.nb_samples - 2*self.x_hist) / (self.stride * self.batch_size))))

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
            batch_x [np.ndarray]: [nb_batches, nb_samples, nb_channels, ...]. Will have trailing dim of size 1 if cut_trailing_dim is False
            batch_y [np.ndarray]: [nb_batches, nb_samples, nb_classes]. nb_classes depends on mode
            weights_y [np.ndarray]: [nb_batches, nb_samples, nb_classes]. nb_classes depends on mode
        """

        batch_x = np.zeros((self.batch_size, self.x_hist, *self.x.shape[1:]), dtype=self.x.dtype)
        
        # HACK TO MAKE THIS WORK WITH pre-processed (multi-frequency), single-channel data
        if self.compute_power:
            batch_x = batch_x[..., 0]

        if self.with_y:
            if self.with_y_hist:
                batch_y = np.zeros((self.batch_size, self.x_hist, self.nb_classes), dtype=self.y.dtype)
            else:
                batch_y = np.zeros((self.batch_size, self.nb_classes), dtype=self.y.dtype)

        if self.shuffle:
            pts = np.random.randint(0, (self.nb_samples - self.x_hist) / self.stride, self.batch_size)
        else:
            pts = range(idx * self.batch_size, (idx + 1) * self.batch_size, 1)
        
        for cnt, bat in enumerate(pts):
            tmp_x = self.x[bat * self.stride:bat * self.stride + self.x_hist, ...]
            # HACK TO MAKE THIS WORK WITH pre-processed (multi-frequency), single-channel data
            if self.compute_power:
                tmp_x = np.sqrt(np.sum(tmp_x**2, axis=-1))
            batch_x[cnt, ...] = tmp_x
            if self.with_y:
                if self.with_y_hist:
                    tmp_y = self.y[bat * self.stride:bat * self.stride + self.x_hist, ...].copy()
                else:
                    tmp_y = self.y[bat * self.stride + self.y_offset, ...].copy()
                tmp_y = self._merge_labels(tmp_y)
                tmp_y[..., 0] = 1.0 - np.sum(tmp_y[..., 1:], axis=-1)
                batch_y[cnt, ...] = tmp_y

        if self.with_y:
            out = [batch_x, batch_y]
            if self.data_padding:
                out.append(self.weights)
        else:
            out = [batch_x, ]
        return out

    def _merge_labels(self, tmp):
        """Merge label types."""
        return merge_labels(tmp, self.mode)
