"""General utilities"""

import tensorflow.keras as keras
import logging
import time
import numpy as np
import yaml
import h5py
import scipy.signal
from . import kapre
from . import tcn
from . import models
from typing import Dict, Callable, Any, List, Tuple, Optional


def load_model(
    file_trunk: str,
    model_dict: Dict[str, Callable],
    model_ext: str = "_model.h5",
    params_ext: str = "_params.yaml",
    compile: bool = True,
    custom_objects: Optional[Dict[str, Callable]] = None,
):
    """Load model with weights.

    First tries to load the full model directly using keras.models.load_model - this will likely fail for models with custom layers.
    Second, try to init model from parameters and then add weights...

    Args:
        file_trunk (str): [description]
        model_dict (Dict[str, Callable): [description]
        model_ext (str, optional): [description]. Defaults to '_weights.h5'.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.
        custom_objects (dict, optional): ...

    Returns:
        keras.Model
    """

    if custom_objects is None:
        custom_objects = {"Spectrogram": kapre.time_frequency.Spectrogram, "TCN": tcn.tcn_new.TCN}

    try:
        model_filename = _download_if_url(file_trunk + model_ext)
        model = keras.models.load_model(model_filename, custom_objects=custom_objects)
    except (SystemError, ValueError, AttributeError):
        logging.debug(
            "Failed to load model using keras, likely because it contains custom layers. Will try to init model architecture from code and load weights from `_model.h5` into it.",
            exc_info=False,
        )
        logging.debug("", exc_info=True)
        model = load_model_from_params(file_trunk, model_dict, weights_ext=model_ext, params_ext=params_ext, compile=compile)
    return model


def load_model_from_params(
    file_trunk: str,
    model_dict: Dict[str, Callable],
    weights_ext: str = "_model.h5",
    params_ext: str = "_params.yaml",
    compile: bool = True,
):
    """Init architecture from code and load model weights into it. Helps with model loading issues across TF versions.

    Args:
        file_trunk (str): [description]
        models_dict ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_model.h5' (use weights from model file).
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.

    Returns:
        keras.Model
    """
    params = load_params(file_trunk, params_ext=params_ext)

    # get the model - calls the function that generates a model with parameters
    model = model_dict[params["model_name"]](**params)
    weights_filename = _download_if_url(file_trunk + weights_ext)
    model.load_weights(weights_filename)

    if compile:
        # Compile with random standard optimizer and loss so we can use the model for prediction
        # Just re-compile the model if you want a particular optimizer and loss.
        model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss="mean_squared_error")
    return model


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
            params = yaml.load(f)
    return params


def load_model_and_params(
    model_save_name, model_dict=models.model_dict, custom_objects=None
) -> Tuple[keras.Model, Dict[str, Any]]:
    """[summary]

    Args:
        model_save_name ([type]): [description]
        model_dict ([type], optional): [description]. Defaults to models.model_dict.
        custom_objects

    Returns:
        keras.Model, Dict[str, Any]: [description]
    """
    params = load_params(model_save_name)
    model = load_model(model_save_name, model_dict=model_dict, custom_objects=custom_objects)
    return model, params


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


def resample(x: np.array, fs_audio: float, fs_model: float):
    """Resample audio to model rate.

    Rounds rates to next even number for efficiency.

    Args:
        x (np.array): _description_
        fs_audio (float): _description_
        fs_model (float): _description_

    Returns:
        np.array: Audio resample to fs_model.
    """
    fs_audio_even = int(fs_audio // 2) * 2
    fs_model_even = int(fs_model // 2) * 2
    gcd = np.gcd(fs_audio_even, fs_model_even)
    x = scipy.signal.resample_poly(x, fs_audio_even // gcd, fs_model_even // gcd, axis=0)
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
