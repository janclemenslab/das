import keras
import numpy as np

from das.models.kapre.time_frequency import Spectrogram


def test_trainable_spectrogram_handles_silent_frames():
    inputs = keras.Input(shape=(128, 1))
    spectrogram = Spectrogram(
        n_dft=64,
        n_hop=16,
        power_spectrogram=1.0,
        return_decibel_spectrogram=True,
        trainable_kernel=True,
    )
    model = keras.Model(inputs, spectrogram(inputs))
    model.compile(optimizer="adam", loss="mse")

    x = np.zeros((1, 128, 1), dtype=np.float32)
    x[:, 64:, 0] = np.sin(np.arange(64, dtype=np.float32))
    y = np.zeros((1, *model.output_shape[1:]), dtype=np.float32)
    model.train_on_batch(x, y)

    assert all(np.isfinite(keras.ops.convert_to_numpy(weight)).all() for weight in spectrogram.trainable_weights)
