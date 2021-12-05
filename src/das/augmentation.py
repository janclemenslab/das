"""Waveform level augmentations.

Individual implementations of `Augmentation` are callables that accept
the signal to augment as an argument and return the augmented signal.

Augmentation parameters can be `Constant`, or random with `Normal` or `Uniform` distribution.
Random parameters will be sampled from a given distribution anew for each augmentation.

`aug = Gain(gain=Normal(mean=1, std=0.5)); augmented_signal = aug(signal)`
"""
import numpy as np
from typing import List, Optional, Callable
import scipy.signal


class Param():
    """Base class for all parameters.

    Parameters are callables that return parameter values.
    """
    pass


class Constant(Param):
    """Constant parameter."""

    def __init__(self, value: float = 0.0):
        """
        Args:
            value (float, optional): Constant parameter value. Defaults to 0.0.
        """
        self.value = value

    def __call__(self, shape=1) -> np.ndarray:
        return self.value * np.ones(shape)


class Normal(Param):
    """Normally distributed parameter."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Args:
            mean (float, optional): Mean of the Normal pdf. Defaults to 0.0.
            std (float, optional): Standerd deviation of the Normal pdf. Defaults to 1.0.
        """
        self.mean = mean
        self.std = std

    def __call__(self, shape=1) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size=shape)


class Uniform(Param):
    """Uniformly distributed parameter."""
    def __init__(self, lower: float = -1.0, upper: float = 1.0):
        """
        Args:
            lower (float, optional): Lower bound. Defaults to -1.0.
            upper (float, optional): Upper bound. Defaults to 1.0.
        """
        self.lower = lower
        self.upper = upper

    def __call__(self, shape=1) -> np.ndarray:
        return np.random.uniform(self.lower, self.upper, size=shape)


class Augmentation(Callable):
    """Base class for all augmentations.

    Augmentations are callables that return the augmented input.
    Can optionally pass through a second input."""
    def __call__(self, batch_x, batch_y=None):
        this_batch_x = batch_x.copy()
        for batch in range(batch_x.shape[0]):
            this_batch_x[batch, ...] = self._apply(this_batch_x[batch, ...])
        if batch_y is None:
            return this_batch_x
        else:
            return this_batch_x, batch_y


class Gain(Augmentation):
    """Multiply signal with gain factor."""

    def __init__(self, gain: Param):
        """
        Args:
            gain (Param): Gain.
        """
        self.gain = gain

    def _apply(self, x):
        x *= self.gain()
        return x


class Offset(Augmentation):
    """Add horizontal offset."""

    def __init__(self, offset: Param):
        """
        Args:
            offset (Param): Offset.
        """
        self.offset = offset

    def _apply(self, x):
        x = x + self.offset()
        return x


class HorizontalFlip(Augmentation):
    """Horizontally flip signal."""

    def __init__(self, flip: Param):
        """
        Args:
            flip (Param): Signal is flipped if >0.
        """
        self.flip = flip

    def _apply(self, x):
        if self.flip() > 0:
            x *= -1
        return x


class Upsampling(Augmentation):
    """Upsample signal."""

    def __init__(self, factor: Param):
        """
        Args:
            factor (Param): Upsampling factor.

        Raises:
            ValueError: if 'lower' attr of factor is <1 (would correspond to downsampling).
        """
        if hasattr(factor, 'lower') and factor.lower < 1:
            raise ValueError(f'Factor is {factor} - at the moment only upsampling (factors >= 1) is allowed.')
        self.factor = factor

    def _apply(self, x):
        len_x = x.shape[0]
        x = scipy.signal.resample_poly(x, up=int(len_x * self.factor()), down=len_x)
        x = x[:len_x]
        return x


class MaskNoise(Augmentation):
    """Add noise or replace signal by noise for the full duration or a part of it."""

    def __init__(self, std: Optional[Param] = None, mean: Optional[Param] = None,
                 duration: Optional[Param] = None, add: bool = True):
        """
        Args:
            std (Optional[Param]): std of noise. Defaults to 1.
            mean (Optional[Param]): mean of noise. Defaults to 0.
            duration (Optional[Param]): nb_samples, Optional. If omitted will mask full duration.
            add (bool): add or replace. Defaults to True.
        """
        self.std = std
        if mean is None:
            mean = Constant(0)
        self.mean = mean
        self.duration = duration
        self.add = add

    def _apply(self, x):
        len_x = x.shape[0]
        if self.duration is None:
            mask_start = 0
            duration = len_x
        else:
            duration = int(self.duration())
            mask_start = np.random.randint(low=0, high=len_x - duration)
        noise = np.random.randn(duration) * self.std() + self.mean(shape=(1, *x.shape[1:]))
        if self.add:
            x[mask_start:mask_start + duration] += noise
        else:
            x[mask_start:mask_start + duration] = noise
        return x


class MaskMean(Augmentation):
    """Replaces stretch of `duration` samples with mean over that stretch."""

    def __init__(self, duration: Param):
        """
        Args:
            duration (Param): Duration of the stretch, in samples.
        """
        self.duration = duration

    def _apply(self, x):
        len_x = x.shape[0]
        mask_start = np.random.randint(low=0, high=len_x - self.duration)
        x[mask_start:mask_start + self.duration, :] = np.mean(x[mask_start:mask_start + self.duration], axis=0)
        return x


class CircShift(Augmentation):
    """Circularly shift input along the first axis."""

    def __init__(self, shift: Param):
        """
        Args:
            shift (Param): Amount of shift, in samples.
        """
        self.shift = shift

    def _apply(self, x):
        x = np.roll(x, self.shift().astype(int), axis=0)
        return x


class Augmentations():
    """Bundles several augmentations."""

    def __init__(self, augmentations: List[Augmentation]):
        """
        Args:
            augmentations (List[Augmentation]): List of augmentations.
        """
        self.augmentations = augmentations

    def __call__(self, batch_x, batch_y=None):
        for augmentation in self.augmentations:
            batch_x = augmentation(batch_x)
        if batch_y is None:
            return batch_x
        else:
            return batch_x, batch_y
