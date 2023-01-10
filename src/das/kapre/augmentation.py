# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class AdditiveNoise(Layer):
    """Adds Gaussian noise to the spectrogram."""

    def __init__(self, power: float = 0.1, random_gain: True = False, noise_type: str = "white", **kwargs):
        """Init.

        Args:
            power (float, optional): Standard deviation of the noise. Defaults to 0.1.
            random_gain (True, optional): If `True`, gain is sampled from `uniform(low=0.0, high=power)` in every batch. Defaults to False.
            noise_type (str, optional): Only supports white. Defaults to 'white'.
        """
        assert noise_type in ["white"]
        self.supports_masking = True
        self.power = power
        self.random_gain = random_gain
        self.noise_type = noise_type
        self.uses_learning_phase = True
        super(AdditiveNoise, self).__init__(**kwargs)

    def call(self, x):
        if self.random_gain:
            power = np.random.uniform(0.0, self.power)
        else:
            power = self.power
        noise_x = x + K.random_normal(shape=K.shape(x), mean=0.0, stddev=power)
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {"power": self.power, "random_gain": self.random_gain, "noise_type": self.noise_type}
        base_config = super(AdditiveNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
