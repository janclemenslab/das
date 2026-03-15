"""Model loading utilities."""

import logging
from typing import Callable, Dict, Optional, Tuple

import keras

from .. import utils
from .architectures import model_dict as default_model_dict
from .kapre.time_frequency import Spectrogram
from .tcn.tcn_new import TCN as TCNNew

DEFAULT_CUSTOM_OBJECTS = {"Spectrogram": Spectrogram, "TCN": TCNNew}


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
        custom_objects = DEFAULT_CUSTOM_OBJECTS.copy()

    try:
        model_filename = utils._download_if_url(file_trunk + model_ext)
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
    params = utils.load_params(file_trunk, params_ext=params_ext)

    model = model_dict[params["model_name"]](**params)
    weights_filename = utils._download_if_url(file_trunk + weights_ext)
    model.load_weights(weights_filename, skip_mismatch=True, by_name=True)

    if compile:
        # Compile with random standard optimizer and loss so we can use the model for prediction
        # Just re-compile the model if you want a particular optimizer and loss.
        model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss="mean_squared_error")
    return model


def load_model_and_params(
    model_save_name, model_dict=default_model_dict, custom_objects=None
) -> Tuple[keras.Model, Dict[str, object]]:
    """Load model and parameter dictionary."""
    params = utils.load_params(model_save_name)
    model = load_model(model_save_name, model_dict=model_dict, custom_objects=custom_objects)
    return model, params
