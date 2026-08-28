from pathlib import Path

import h5py
import numpy as np
import pytest

from das import models, utils


PARAMS = {
    "model_name": "tcn_stft",
    "nb_freq": 1,
    "nb_classes": 2,
    "nb_hist": 8,
    "nb_filters": 2,
    "kernel_size": 2,
    "nb_conv": 1,
    "dilations": [1],
    "nb_pre_conv": 1,
    "pre_nb_dft": 4,
    "compile": False,
}


@pytest.mark.parametrize(
    "relative_trunk",
    [
        "docs/tutorials/models/dmel_single_rt/20200430_201821",
        "docs/tutorials/models/dmel_all/20200507_173738",
    ],
)
def test_real_tensorflow_model_loads_and_predicts(relative_trunk):
    trunk = Path(__file__).parents[1] / relative_trunk
    if not all(Path(str(trunk) + suffix).exists() for suffix in ("_params.yaml", "_model.h5")):
        pytest.skip("optional legacy model fixture is not checked into Git")

    model, params = models.load_model_and_params(str(trunk))
    input_shape = [1]
    for axis, size in enumerate(model.input_shape[1:], start=1):
        if size is not None:
            input_shape.append(size)
        elif axis == 1:
            input_shape.append(max(int(params.get("nb_hist", 1024)), 2048))
        else:
            raise AssertionError(f"Cannot infer input shape {model.input_shape}")

    prediction = model.predict(np.zeros(input_shape, dtype=np.float32), verbose=0)

    assert np.isfinite(prediction).all()


def _model_with_known_weights():
    model = models.tcn_stft(**PARAMS)
    for index, variable in enumerate(model.weights, start=1):
        values = np.arange(np.prod(variable.shape), dtype=np.float32).reshape(variable.shape) / index
        variable.assign(values)
    return model


def _save_legacy_h5_weights(model, filename):
    with h5py.File(filename, "w") as file:
        group = file.create_group("model_weights")
        layer_names = []
        for layer_index, layer in enumerate(layer for layer in model.layers if layer.weights):
            layer_name = f"legacy_layer_{layer_index}"
            layer_names.append(layer_name)
            layer_group = group.create_group(layer_name)
            weight_names = []
            for weight_index, value in enumerate(layer.get_weights()):
                weight_name = f"weight_{weight_index}"
                weight_names.append(weight_name)
                layer_group.create_dataset(weight_name, data=value)
            layer_group.attrs["weight_names"] = np.asarray(weight_names, dtype="S")
        group.attrs["layer_names"] = np.asarray(layer_names, dtype="S")


def test_load_legacy_h5_by_strict_layer_order(tmp_path):
    trunk = str(tmp_path / "legacy")
    expected = _model_with_known_weights()
    utils.save_params(PARAMS, trunk)
    _save_legacy_h5_weights(expected, trunk + "_model.h5")

    loaded, params = models.load_model_and_params(trunk)

    for expected_weight, loaded_weight in zip(expected.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(loaded_weight, expected_weight)
    assert params == PARAMS


def test_load_new_keras_model(tmp_path):
    trunk = str(tmp_path / "new")
    expected = _model_with_known_weights()
    utils.save_params(PARAMS, trunk)
    expected.save(trunk + "_model.keras")

    loaded, params = models.load_model_and_params(trunk)

    for expected_weight, loaded_weight in zip(expected.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(loaded_weight, expected_weight)
    assert params == PARAMS
