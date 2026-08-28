import numpy as np
import keras


def test_legacy_kapre_noise_layer():
    from das.kapre.augmentation import AdditiveNoise

    x = np.ones((2, 4, 1), dtype=np.float32)
    layer = AdditiveNoise(power=0.1)

    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer(x, training=False)), x)
    assert tuple(layer(x, training=True).shape) == x.shape


def test_legacy_losses():
    from das.loss import TMSE, WeightedLoss

    y = np.asarray([[[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]], dtype=np.float32)
    tmse = TMSE(batch_size=1)

    assert tuple(tmse.call(y, y).shape) == (1, 3)
    combined = WeightedLoss([lambda true, pred: pred], [2.0])
    np.testing.assert_allclose(keras.ops.convert_to_numpy(combined.call(y, y)), 2 * y)
