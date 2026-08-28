from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from das import models
from das.utils import load_model_and_params, save_params


@pytest.mark.parametrize(
    "relative_trunk",
    [
        "docs/tutorials/models/dmel_single_rt/20200430_201821",
        "docs/tutorials/models/dmel_all/20200507_173738",
    ],
)
def test_legacy_model_loads_and_predicts(relative_trunk):
    trunk = Path(__file__).parents[1] / relative_trunk
    if not all(Path(str(trunk) + suffix).exists() for suffix in ("_params.yaml", "_model.h5")):
        pytest.skip("optional legacy model fixture is not checked into Git")
    model, params = load_model_and_params(str(trunk))
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


def test_model_train_save_reload_round_trip(tmp_path):
    np.random.seed(0)
    tf.random.set_seed(0)
    params = {
        "model_name": "tcn",
        "nb_freq": 1,
        "nb_classes": 2,
        "nb_hist": 64,
        "nb_filters": 2,
        "kernel_size": 3,
        "nb_conv": 1,
        "dilations": [1],
        "morph_nb_kernels": 0,
        "learning_rate": 0.001,
    }
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(4, 64, 1)).astype(np.float32)
    labels = tf.one_hot((samples[..., 0] > 0).astype(np.int32), depth=2).numpy()
    model = models.model_dict[params["model_name"]](**params)
    history = model.fit(samples, labels, epochs=1, batch_size=2, verbose=0)
    prediction_before = model.predict(samples, verbose=0)
    trunk = tmp_path / "tiny"
    model.save(str(trunk) + "_model.h5")
    save_params(params, str(trunk))

    reloaded, _ = load_model_and_params(str(trunk))
    prediction_after = reloaded.predict(samples, verbose=0)

    assert np.isfinite(history.history["loss"][0])
    np.testing.assert_allclose(prediction_after, prediction_before, rtol=1e-5, atol=1e-6)
