"""Model loading utilities."""

from pathlib import Path
import logging
from typing import Callable, Dict, Optional, Tuple

import h5py
import keras
import numpy as np

from .. import utils
from .architectures import model_dict as default_model_dict
from .kapre.time_frequency import Spectrogram
from .tcn.tcn_new import TCN as TCNNew

DEFAULT_CUSTOM_OBJECTS = {
    "Spectrogram": Spectrogram,
    "TCN": TCNNew,
}


def _resolve_model_ext(file_trunk: str, model_ext: Optional[str]) -> str:
    if model_ext is not None:
        return model_ext
    if not file_trunk.startswith("http") and Path(file_trunk + "_model.keras").exists():
        return "_model.keras"
    return "_model.h5"


def _load_legacy_h5_weights(model: keras.Model, filename: str) -> None:
    """Strictly load legacy Keras HDF5 weights in saved layer order."""
    with h5py.File(filename, "r") as file:
        group = file["model_weights"] if "model_weights" in file else file
        saved_layers = []
        for raw_layer_name in group.attrs.get("layer_names", []):
            layer_name = raw_layer_name.decode() if isinstance(raw_layer_name, bytes) else raw_layer_name
            layer_group = group[layer_name]
            values = []
            for raw_weight_name in layer_group.attrs.get("weight_names", []):
                weight_name = raw_weight_name.decode() if isinstance(raw_weight_name, bytes) else raw_weight_name
                values.append(np.asarray(layer_group[weight_name]))
            if values:
                saved_layers.append((layer_name, values))

    model_layers = [layer for layer in model.layers if layer.weights]
    if len(saved_layers) != len(model_layers):
        raise ValueError(f"Legacy model has {len(saved_layers)} weighted layers; expected {len(model_layers)}.")

    for (saved_name, values), layer in zip(saved_layers, model_layers):
        saved_shapes = [value.shape for value in values]
        model_shapes = [tuple(weight.shape) for weight in layer.weights]
        if saved_shapes != model_shapes:
            raise ValueError(
                f"Legacy layer {saved_name!r} has weight shapes {saved_shapes}; " f"{layer.name!r} expects {model_shapes}."
            )
        layer.set_weights(values)


def load_model(
    file_trunk: str,
    model_dict: Dict[str, Callable] = default_model_dict,
    model_ext: Optional[str] = None,
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
        model_ext (str, optional): Model suffix. Auto-detects `_model.keras`
            locally and otherwise defaults to legacy `_model.h5`.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.
        custom_objects (dict, optional): ...

    Returns:
        keras.Model
    """
    if custom_objects is None:
        custom_objects = DEFAULT_CUSTOM_OBJECTS.copy()

    model_ext = _resolve_model_ext(file_trunk, model_ext)
    try:
        model_filename = utils._download_if_url(file_trunk + model_ext)
        model = keras.models.load_model(model_filename, custom_objects=custom_objects, compile=compile)
    except (SystemError, ValueError, AttributeError, EOFError, TypeError):
        logging.debug(
            "Failed to load the serialized model. Rebuilding it from parameters and loading weights.",
            exc_info=False,
        )
        logging.debug("", exc_info=True)
        model = load_model_from_params(file_trunk, model_dict, weights_ext=model_ext, params_ext=params_ext, compile=compile)
    return model


def load_model_from_params(
    file_trunk: str,
    model_dict: Dict[str, Callable] = default_model_dict,
    weights_ext: Optional[str] = None,
    params_ext: str = "_params.yaml",
    compile: bool = True,
):
    """Init architecture from code and load model weights into it. Helps with model loading issues across TF versions.

    Args:
        file_trunk (str): [description]
        models_dict ([type]): [description]
        weights_ext (str, optional): Model suffix. Auto-detects `_model.keras`
            locally and otherwise defaults to legacy `_model.h5`.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.

    Returns:
        keras.Model
    """
    params = utils.load_params(file_trunk, params_ext=params_ext)

    def build_model():
        return model_dict[params["model_name"]](**params)

    model = build_model()
    weights_ext = _resolve_model_ext(file_trunk, weights_ext)
    weights_filename = utils._download_if_url(file_trunk + weights_ext)
    try:
        model.load_weights(weights_filename, skip_mismatch=False, by_name=False)
    except (ValueError, TypeError) as strict_error:
        model = build_model()
        try:
            if not weights_ext.endswith(".h5"):
                raise strict_error
            _load_legacy_h5_weights(model, weights_filename)
        except (ValueError, KeyError, OSError):
            logging.warning(
                "Strict weight loading failed for %s (%s). Retrying by name and skipping mismatches.",
                weights_filename,
                strict_error,
            )
            model = build_model()
            if weights_ext.endswith(".h5"):
                model.load_weights(weights_filename, skip_mismatch=True, by_name=True)
            else:
                model.load_weights(weights_filename, skip_mismatch=True)

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
