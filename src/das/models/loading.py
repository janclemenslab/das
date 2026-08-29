"""Model loading utilities."""

import json
from pathlib import Path
import logging
from typing import Callable, Dict, Optional, Tuple

import h5py
import keras
import numpy as np

from .. import utils
from .architectures import model_dict as default_model_dict
from .kapre import backend as kapre_backend
from .kapre.time_frequency import Spectrogram
from .tcn.tcn_new import TCN as TCNNew


class _LegacySlicingOpLambda(keras.layers.Layer):
    """Keras 3 replacement for TensorFlow's serialized ``x[..., None]`` op."""

    def call(self, inputs):
        return keras.ops.expand_dims(inputs, axis=-1)


DEFAULT_CUSTOM_OBJECTS = {
    "Spectrogram": Spectrogram,
    "SlicingOpLambda": _LegacySlicingOpLambda,
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
            weight_names = []
            values = []
            for raw_weight_name in layer_group.attrs.get("weight_names", []):
                weight_name = raw_weight_name.decode() if isinstance(raw_weight_name, bytes) else raw_weight_name
                weight_names.append(weight_name.removesuffix(":0"))
                values.append(np.asarray(layer_group[weight_name]))
            if values:
                saved_layers.append((layer_name, weight_names, values))

    model_layers = [layer for layer in model.layers if layer.weights]
    if len(saved_layers) != len(model_layers):
        raise ValueError(f"Legacy model has {len(saved_layers)} weighted layers; expected {len(model_layers)}.")

    for (saved_name, weight_names, values), layer in zip(saved_layers, model_layers):
        model_weight_names = [getattr(weight, "path", weight.name).removesuffix(":0") for weight in layer.weights]
        if len(set(weight_names)) == len(weight_names) and set(weight_names) == set(model_weight_names):
            values_by_name = dict(zip(weight_names, values))
            values = [values_by_name[name] for name in model_weight_names]
        saved_shapes = [value.shape for value in values]
        model_shapes = [tuple(weight.shape) for weight in layer.weights]
        if saved_shapes != model_shapes:
            raise ValueError(
                f"Legacy layer {saved_name!r} has weight shapes {saved_shapes}; " f"{layer.name!r} expects {model_shapes}."
            )
        layer.set_weights(values)


def _migrate_legacy_h5_config(value):
    """Translate the small set of TensorFlow-Keras 2 config differences seen in legacy models."""
    if isinstance(value, str):
        return value.replace("/", "_")
    if isinstance(value, list):
        return [_migrate_legacy_h5_config(item) for item in value]
    if not isinstance(value, dict):
        return value

    migrated = {key: _migrate_legacy_h5_config(item) for key, item in value.items()}
    if migrated.get("class_name") == "DepthwiseConv2D":
        migrated["config"].pop("groups", None)
    elif migrated.get("class_name") == "TimeDistributed":
        migrated["config"]["layer"] |= {"module": "keras.layers", "registered_name": None}
    elif migrated.get("class_name") == "SlicingOpLambda":
        migrated["config"].pop("function", None)
        migrated["inbound_nodes"] = [[[node[0], node[1], node[2], {}]] for node in migrated["inbound_nodes"]]

    layers = migrated.get("config", {}).get("layers", [])
    nested_models = {layer.get("name") for layer in layers if layer.get("class_name") == "Functional"}

    def shift_nested_model_nodes(item):
        if isinstance(item, list):
            if len(item) >= 4 and isinstance(item[0], str) and item[0] in nested_models and isinstance(item[1], int):
                item[1] = max(0, item[1] - 1)
            else:
                for child in item:
                    shift_nested_model_nodes(child)
        elif isinstance(item, dict):
            for child in item.values():
                shift_nested_model_nodes(child)

    for layer in layers:
        shift_nested_model_nodes(layer.get("inbound_nodes", []))
    return migrated


def _load_migrated_legacy_h5_model(filename: str, custom_objects: Dict[str, Callable], compile: bool) -> keras.Model:
    from keras.src.legacy.saving import saving_utils

    with h5py.File(filename, "r") as file:
        config = _migrate_legacy_h5_config(json.loads(file.attrs["model_config"]))
    model = saving_utils.model_from_config(config, custom_objects=custom_objects)
    _load_legacy_h5_weights(model, filename)
    if compile:
        model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss="mean_squared_error")
    return model


def _repair_nonfinite_spectrogram_kernels(model: keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, Spectrogram) and not all(np.isfinite(np.asarray(weight)).all() for weight in layer.weights):
            logging.warning("Resetting non-finite STFT kernels in legacy layer %s.", layer.name)
            real, imag = kapre_backend.get_stft_kernels(layer.n_dft)
            layer.dft_real_kernels.assign(real)
            layer.dft_imag_kernels.assign(imag)
        elif isinstance(layer, keras.Model):
            _repair_nonfinite_spectrogram_kernels(layer)


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
    custom_objects = DEFAULT_CUSTOM_OBJECTS | (custom_objects or {})

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
        params = utils.load_params(file_trunk, params_ext=params_ext)
        if model_ext.endswith(".h5") and "model_name" not in params:
            model = _load_migrated_legacy_h5_model(model_filename, custom_objects, compile)
        else:
            model = load_model_from_params(
                file_trunk, model_dict, weights_ext=model_ext, params_ext=params_ext, compile=compile
            )
    _repair_nonfinite_spectrogram_kernels(model)
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
