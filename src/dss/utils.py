"""General utilities"""
import tensorflow.keras as keras
import logging
import time
import numpy as np
import yaml
import pywt
import skimage.util.shape
import scipy.signal
import scipy.ndimage
import h5py


class LossHistory(keras.callbacks.Callback):
    """[summary]

    Args:
        keras ([type]): [description]
    """

    def __init__(self):
        self.min_loss = np.inf

    def on_train_begin(self, logs):
        logging.info('training started')

    def on_epoch_begin(self, epoch, logs):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs):
        duration = time.time() - self.t0
        current_loss = logs.get('val_loss')
        current_train_loss = logs.get('loss')
        self.min_loss = np.min((current_loss, self.min_loss))
        logging.info('ep{0:>5d} ({1:1.4f} seconds): train_loss={4:1.5f} val_loss= {2:1.5f}/{3:1.5f} (current/min)'.format(epoch, duration, current_loss, self.min_loss, current_train_loss))

    def on_training_end(self, logs):
        logging.info('trained ended')


def save_model(model, file_trunk, weights_ext='_weights.h5', architecture_ext='_arch.yaml'):
    """Save model weights and architecture to separate files.

    Args:
        model ([type]): [description]
        file_trunk ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_weights.h5'.
        architecture_ext (str, optional): [description]. Defaults to '_arch.yaml'.
    """
    save_model_architecture(model, file_trunk, architecture_ext)
    model.save_weights(file_trunk + weights_ext)


def save_model_architecture(model, file_trunk, architecture_ext='_arch.yaml'):
    """Save model architecture as yaml to separate files.

    Args:
        model ([type]): [description]
        file_trunk ([type]): [description]
        architecture_ext (str, optional): [description]. Defaults to '_arch.yaml'.
    """
    with open(file_trunk + architecture_ext, 'w') as f:
        f.write(model.to_yaml())


def load_model(file_trunk, model_dict, weights_ext='_weights.h5', from_epoch=False,
               params_ext='_params.yaml', compile=True):
    """Load model.

    First tries to load the full model directly using keras.models.load_model - this will likely fail for models with custom layers.
    Second, try to init model from parameters and then add weights...

    Args:
        file_trunk ([type]): [description]
        model_dict ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_weights.h5'.
        from_epoch ([type], optional): [description]. Defaults to None.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    if from_epoch:
        file_trunk_params = file_trunk[:-4]  # remove epoch number from params file name
        weights_ext = file_trunk[-4:] + '_weights.h5'  # add epoch number to weight file to load epoch specific weights
        model_ext = '_weights.h5'
    else:
        file_trunk_params = file_trunk  # remove epoch number from params file name
        weights_ext = '_model.h5'  # add epoch number to weight file to load epoch specific weights
        model_ext = '_model.h5'

    try:
        model = keras.models.load_model(file_trunk + model_ext)
    except SystemError:
        logging.debug('Failed to load model using keras, likely because it contains custom layers. Will try to init model architecture from code and load weights into it.', exc_info=False)
        logging.debug('', exc_info=True)
        model = load_model_from_params(file_trunk_params, model_dict, weights_ext, compile=compile)
    return model


def load_model_from_params(file_trunk, models_dict, weights_ext='_weights.h5', params_ext='_params.yaml', compile=True):
    """Load model weights and architecture from separate files.

    Args:
        file_trunk ([type]): [description]
        models_dict ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_weights.h5'.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    params = load_params(file_trunk, params_ext)
    model = models_dict[params['model_name']](**params)  # get the model - calls the function that generates a model with parameters
    model.load_weights(file_trunk + weights_ext)
    if compile:
        # Compile with random standard optimizer and loss so we can use the model for prediction
        # Just re-compile the model if you want a particular optimizer and loss.
        model.compile(optimizer=keras.optimizers.Adam(amsgrad=True),
                      loss="mean_squared_error")
    return model


def save_params(params, file_trunk, params_ext='_params.yaml'):
    """Save model/training parameters to yaml.

    Args:
        params ([type]): [description]
        file_trunk ([type]): [description]
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
    """
    with open(file_trunk + params_ext, 'w') as f:
        yaml.dump(params, f)


def load_params(file_trunk, params_ext='_params.yaml'):
    """Load model/training parameters from yaml

    Args:
        file_trunk ([type]): [description]
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.

        Returns:
        [type]: [description]
    """
    with open(file_trunk + params_ext, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            params = yaml.load(f)
    return params


def load_from(filename, datasets):
    """Load datasets from h5 file.

    Args:
        filename ([type]): [description]
        datasets ([type]): [description]

        Returns:
        [type]: [description]
    """
    data = dict()
    with h5py.File(filename, 'r') as f:
        data = {dataset: f[dataset][:] for dataset in datasets}
    return data

# move to .data?
def smooth_labels(labels, pulse_times, s0=None, s1=None, buf=50):
    """[summary]

    Args:
        labels ([type]): [description]
        pulse_times ([type]): [description]
        s0 ([type]): [description]
        s1 ([type]): [description]
        buf (int, optional): [description]. Defaults to 50.

        Returns:
        [type]: [description]
    """
    pulse_times = pulse_times.T
    if s0 is None:
        condition = np.concatenate((pulse_times < s1 - buf - 1, pulse_times > s0 + buf + 1), axis=0)
        within_range_pulses = np.all(condition, axis=0)
        pulse_times = pulse_times[within_range_pulses, 0] - s0
    labels[:, 1] = 0
    for stp in range(-buf, buf):
        labels[np.uintp(pulse_times + stp), 1] = 1

    labels[labels[:, 1] == 1, 0] = 0  # pulse excludes sine
    labels[labels[:, 1] == 1, 2] = 0  # pulse excludes silence
    labels[np.sum(labels, axis=1) == 0, 0] = 1
    return labels


def running_fun(x, fun, winlen, stride=1, window=None):
    """[summary]

    Args:
        x ([type]): [description]
        fun ([type]): [description]
        winlen ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.
        window ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if len(x.shape) == 1:
        x = x[..., np.newaxis]
    X = skimage.util.shape.view_as_windows(x, [winlen, 1], [stride, 1])[..., 0]
    if window is not None:
        X = X * window
    y = fun(X, axis=2)
    return y


def cwt(dta, scales=range(8, 150, 2), wavelet_name='cmor1.5-1.0', stack_imre=True, power_only=False):
    """Cont Wvlt Trsfrm.

    Args:
        dta: [T,]
        scales=range(8, 150, 2)
        wavelet_name='cmor1.5-1.0'
        stack_imre=True
        power_only=False (stack_imre is ignored if power_only is True)
    """
    # frequencies = pywt.scale2frequency(wavelet_name, scales ) / dt
    # logging.info(f'freq range {np.min(frequencies):1.1f}-{np.max(frequencies):1.1f}Hz')
    if dta.ndim == 2:
        dta = dta[:, 0]
    coef, _ = pywt.cwt(dta, scales, wavelet_name)
    if stack_imre and not power_only:
        coef = np.concatenate((np.real(coef.T), np.imag(coef.T)), axis=1)
    else:
        coef = coef.T

    if power_only:
        coef = np.absolute(coef)
    return coef

# move elsewhere - don't need thos here
def merge_channels(data, sampling_rate, max_filter_len=101, passband=(25, 1500)):
    """Merge multi-channel recording into a single channel based in max amplitude. 

    Args:
        data ([type]): [time, channels]
        sampling_rate ([type]): Hz
        max_filter_len (int, optional): in samples. Defaults to 101.
        passband (tuple, optional): (fmin, fmax) Hz. Defaults to (25, 1500).

    Returns:
        merged channels [time, 1]
    """
    # TODO Should use segmentation data (y_pred?) to make sure we get all song
    # remove all nan/inf data
    mask = ~np.isfinite(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    # band-pass filter out noise on each channel
    b, a = scipy.signal.butter(6, passband, btype='bandpass', fs=sampling_rate)
    data = scipy.signal.filtfilt(b, a, data, axis=0, method='pad')
    # find loudest channel in 101-sample windows
    sng_max = scipy.ndimage.maximum_filter1d(np.abs(data), size=max_filter_len, axis=0)
    loudest_channel = np.argmax(sng_max, axis=-1)
    # get linear index and merge channels
    idx = np.ravel_multi_index((np.arange(sng_max.shape[0]), loudest_channel), data.shape)
    data_merged_max = data.ravel()[idx]
    data_merged_max = data_merged_max[:, np.newaxis]  # shape needs to be [nb_samples, 1]
    return data_merged_max


class Timer:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.verbose:
            print(self)

    def __str__(self):
        if self.start is None:
            s = "Timer not started yet."
        elif self.end is None:
            s = "Timer still running."
        elif self.elapsed is not None:
            s = f'Time elapsed {self.elapsed:1.2f} seconds.'
        else:
            s = "Timer in unexpected state."
        return s
