import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from das import models, utils
from das.models import loading


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


def _assert_model_predicts(model, params):
    input_shape = [1]
    for axis, size in enumerate(model.input_shape[1:], start=1):
        if size is not None:
            input_shape.append(size)
        elif axis == 1:
            input_shape.append(max(int(params.get("nb_hist", 1024)), 2048))
        else:
            raise AssertionError(f"Cannot infer input shape {model.input_shape}")

    prediction = model.predict(np.random.default_rng(0).normal(size=input_shape).astype(np.float32), verbose=0)
    assert np.isfinite(prediction).all()


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
    _assert_model_predicts(model, params)


def test_external_legacy_model_catalog():
    catalog = os.environ.get("DAS_MODEL_CATALOG")
    if not catalog:
        pytest.skip("set DAS_MODEL_CATALOG to check an external legacy model collection")

    trunks = [
        str(model_file).removesuffix("_model.h5")
        for model_file in sorted(Path(catalog).rglob("*_model.h5"))
        if Path(str(model_file).removesuffix("_model.h5") + "_params.yaml").exists()
    ]
    assert trunks, f"No model/parameter pairs found under {catalog}"
    for trunk in trunks:
        try:
            model, params = models.load_model_and_params(trunk)
            _assert_model_predicts(model, params)
        except Exception as error:
            pytest.fail(f"Legacy model failed: {trunk}: {error!r}")


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


def test_migrate_tensorflow_keras_h5_config():
    config = {
        "class_name": "Functional",
        "config": {
            "layers": [
                {"class_name": "Functional", "name": "nested/model", "config": {"layers": []}},
                {
                    "class_name": "SlicingOpLambda",
                    "config": {"function": "__operators__.getitem", "name": "slice/op"},
                    "inbound_nodes": [["input", 0, 0, {"slice_spec": [{"class_name": "__ellipsis__"}, None]}]],
                },
                {"class_name": "DepthwiseConv2D", "config": {"groups": 1}},
                {
                    "class_name": "TimeDistributed",
                    "config": {"layer": {"class_name": "Dense", "config": {"units": 2}}},
                },
                {"class_name": "Dense", "inbound_nodes": [["nested/model", 1, 0, {}]]},
            ]
        },
    }

    migrated = loading._migrate_legacy_h5_config(config)
    layers = migrated["config"]["layers"]

    assert layers[0]["name"] == "nested_model"
    assert layers[1]["config"] == {"name": "slice_op"}
    assert layers[1]["inbound_nodes"] == [[["input", 0, 0, {}]]]
    assert "groups" not in layers[2]["config"]
    assert layers[3]["config"]["layer"]["module"] == "keras.layers"
    assert layers[4]["inbound_nodes"][0][1] == 0


def test_repair_nonfinite_legacy_spectrogram_kernels():
    model = models.tcn_stft(**PARAMS)
    spectrogram = next(layer for layer in model.layers if isinstance(layer, loading.Spectrogram))
    spectrogram.dft_real_kernels.assign(np.full(spectrogram.dft_real_kernels.shape, np.nan))
    spectrogram.dft_imag_kernels.assign(np.full(spectrogram.dft_imag_kernels.shape, np.nan))

    loading._repair_nonfinite_spectrogram_kernels(model)

    assert all(np.isfinite(np.asarray(weight)).all() for weight in spectrogram.weights)
