"""Legacy Kapre augmentation layers."""

import numpy as np
import keras


class AdditiveNoise(keras.layers.Layer):
    """Add Gaussian noise while training."""

    def __init__(self, power: float = 0.1, random_gain: bool = False, noise_type: str = "white", **kwargs):
        if noise_type != "white":
            raise ValueError("Only white noise is supported.")
        super().__init__(**kwargs)
        self.supports_masking = True
        self.power = power
        self.random_gain = random_gain
        self.noise_type = noise_type

    def call(self, x, training=None):
        if not training:
            return x
        power = np.random.uniform(0.0, self.power) if self.random_gain else self.power
        return x + keras.random.normal(keras.ops.shape(x), mean=0.0, stddev=power)

    def get_config(self):
        return {
            **super().get_config(),
            "power": self.power,
            "random_gain": self.random_gain,
            "noise_type": self.noise_type,
        }
