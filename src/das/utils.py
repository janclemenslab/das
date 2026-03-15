"""General utilities"""

import keras as keras
import time
import numpy as np
import yaml
import h5py
import scipy.signal
from typing import Dict, Any, List, Optional


def save_params(params: Dict[str, Any], file_trunk: str, params_ext: str = "_params.yaml"):
    """Save model/training parameters to yaml.

    Args:
        params (Dict[str]): [description]
        file_trunk (str): [description]
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
    """
    with open(file_trunk + params_ext, "w") as f:
        yaml.dump(params, f)


def load_params(file_trunk: str, params_ext: str = "_params.yaml") -> Dict[str, Any]:
    """Load model/training parameters from yaml

    Args:
        file_trunk (str): [description]
        params_ext (strs, optional): [description]. Defaults to '_params.yaml'.

    Returns:
        Dict[str, Any]: Parameter dictionary
    """
    filename = _download_if_url(file_trunk + params_ext)
    with open(filename, "r") as f:
        try:
            params = yaml.unsafe_load(f)
        except AttributeError:
            params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def _download_if_url(url: str):
    if not url.startswith("http"):
        return url
    else:
        import urllib.request
        import tempfile
        from pathlib import Path

        filename = url.split("/")[-1]  # get filename
        tmpdir = tempfile.mkdtemp()
        local_path = Path(tmpdir) / filename
        urllib.request.urlretrieve(url, local_path)
        return local_path


def load_from(filename: str, datasets: List[str]):
    """Load datasets from h5 file.

    Args:
        filename (str)
        datasets (List[str]): Names of the datasets (=keys) to load

    Returns:
        [type]: [description]
    """
    data = dict()
    with h5py.File(filename, "r") as f:
        data = {dataset: f[dataset][:] for dataset in datasets}
    return data


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
            s = f"Time elapsed {self.elapsed:1.2f} seconds."
        else:
            s = "Timer in unexpected state."
        return s


class QtProgressCallback(keras.callbacks.Callback):
    def __init__(self, nb_epochs, comms):
        """Init the callback.

        Args:
            nb_epochs ([type]): number of training epochs
            comms (tuple): tuple of (multiprocessing.Queue, threading.Event)
                The queue is used to transmit progress updates to the GUI,
                the event is set in the GUI to stop training.
        """
        super().__init__()
        self.nb_epochs = nb_epochs
        self.queue = comms[0]
        self.stop_event = comms[1]

    def _check_if_stopped(self):
        try:
            if self.stop_event.is_set():
                self.model.stop_training = True
        except Exception as e:
            print(e)

    def on_train_begin(self, logs=None):
        self.queue.put((0, "Starting training."))

    def on_train_end(self, logs=None):
        self.queue.put((-1, "Finishing training."))

    def on_epoch_end(self, epoch, logs=None):
        self.queue.put((epoch, f"Epoch {epoch}/{self.nb_epochs}"))

    def on_train_batch_end(self, batch, logs=None):
        self._check_if_stopped()

    def on_test_batch_end(self, batch, logs=None):
        self._check_if_stopped()

    def on_predict_batch_end(self, batch, logs=None):
        self._check_if_stopped()


def resample(x: np.ndarray, fs_audio: float, fs_target: float) -> np.ndarray:
    """Resample source to target rate along axis 0.

    Rounds rates to next even number for efficiency.

    Args:
        x (np.ndarray): [time x channels] array.
        fs_audio (float): Hz.
        fs_target (float): Hz.

    Returns:
        np.ndarray: Resampled audio.
    """
    fs_audio_even = int(fs_audio // 2) * 2
    fs_target_even = int(fs_target // 2) * 2
    gcd = np.gcd(fs_audio_even, fs_target_even)
    x = scipy.signal.resample_poly(x, fs_target_even // gcd, fs_audio_even // gcd, axis=0)
    return x


def bandpass_filter_song(
    x: np.ndarray, sampling_rate_hz: float, f_low: Optional[float] = None, f_high: Optional[float] = None
) -> np.ndarray:
    """Band-pass filter channel data

    Args:
        x (np.ndarray): Audio data[T,] or [T, nb_channels]
        sampling_rate_hz (float): Sampling rate in Hz
        f_low (Optional[float], optional): Lower cutoff in Hz. Defaults to 1.0 (None).
        f_high (Optional[float], optional): Upper cutoff in Hz. Defaults to sampling_rate_hz/2 (None).

    Returns:
        np.ndarray: Sampled data
    """
    if f_low is None:
        f_low = 1.0

    if f_high is None:
        f_high = sampling_rate_hz / 2 - 1

    f_high = min(f_high, sampling_rate_hz / 2 - 1)

    sos_bp = scipy.signal.butter(5, [f_low, f_high], "bandpass", output="sos", fs=sampling_rate_hz)
    x = scipy.signal.sosfiltfilt(sos_bp, x, axis=0)
    return x
