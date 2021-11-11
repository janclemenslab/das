import numpy as np
from typing import List
import scipy.signal


class Param():
    pass

class Constant(Param):

    def __init__(self, value: float = 0.0):
        self.value

    def make(self, shape):
        return self.value * np.ones(shape)


class Normal(Param):

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def make(self, shape):
        return np.random.normal(self.mean, self.std, size=shape)


class Uniform(Param):

    def __init__(self, lower: float = -1.0, upper: float = 1.0):
        self.lower = lower
        self.upper = upper

    def make(self, shape):
        return np.random.uniform(self.lower, self.upper, size=shape)


class Augmentation():

    def apply(self, batch_x, batch_y):
        for batch in range(batch_x.shape[0]):
            batch_x[batch, ...] = self._apply(batch_x[batch, ...])
        return batch_x, batch_y


class Gain(Augmentation):

    def __init__(self, gain: Param):
        self.gain = gain

    def _apply(self, x):
        x *= self.gain(shape=(1,))
        return x


class Offset(Augmentation):

    def __init__(self, offset: Param):
        self.offset = offset

    def _apply(self, x):
        x += self.offset(shape=(1,))
        return x


class HorizontalFlip(Augmentation):

    def __init__(self, flip: Param):
        self.flip = flip

    def _apply(self, x):
        if self.flip() > 0:
            x *= -1
        return x


class AddNoise(Augmentation):

    def __init__(self, mean: Param, std: Param):
        self.mean = mean
        self.std = std

    def _apply(self, x):
        x += self.gain(shape=x.shape)
        return x


class Upsampling(Augmentation):

    def __init__(self, factor: Param):
        if factor < 1:
            raise ValueError(f'Factor is {factor} but at the moment only upsampling (factor > 1) is allowd.')
        self.factor = factor

    def _apply(self, x):
        len_x = x.shape[0]
        x = scipy.signal.resample(x, int(len_x * self.factor) )
        x = x[:len_x]
        return x


class MaskNoise(Augmentation):
    # duration, std
    pass


class MaskMean(Augmentation):
    # replace part with avg
    pass


class Augmentations():

    def __init__(self, augmentations: List[Augmentation]):
        self.augmentations = augmentations

    def apply(self, batch_x, batch_y):
        for augmentation in self.augmentations:
            batch_x, batch_y = augmentation.apply(batch_x, batch_y)
        return batch_x, batch_y