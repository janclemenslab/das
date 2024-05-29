"""Waveform level augmentations.

Individual implementations of `Augmentation` are callables that accept
the signal to augment as an argument and return the augmented signal.

Augmentation parameters can be `Constant`, or random with `Normal` or `Uniform` distribution.
Random parameters will be sampled from a given distribution anew for each augmentation.

`aug = Gain(gain=Normal(mean=1, std=0.5)); augmented_signal = aug(signal)`

Can be configured using a yaml file:
```yaml
Gain:  # Name of the augmentation class
  gain:  # arg for the augmentation class
    Uniform:  # Param type
       lower: 0.5  # param args
       upper: 2  # param args

MaskNoise:
  std:
    Normal:
      mean: 0
      std: 0.05
  mean:
    Constant:
       value: 0

NotchFilter:
  freq:  # Param-type arg
    Uniform:
      lower: 100
      upper: 600
  Q: 30  # standard arg
  samplerate_Hz: 10_000  # standard arg
```
Caution: You need to add a suffix starting with '-' (like "MaskNoise-1") to the class name
         if you want to use a class multiple times

`augs = Augmentations.from_yaml(filename)`
"""

import numpy as np
from typing import List, Optional
import scipy.signal
import yaml
from dataclasses import dataclass
from typing import Dict
import logging

logger = logging.getLogger(__name__)

aug_dict = dict()
params_dict = dict()


def _register_augmentation(func):
    """Adds func to model_dict Dict[augname: augfunc]. For selecting augs by string."""
    aug_dict[func.__name__] = func
    return func


def _register_param(func):
    """Adds func to model_dict Dict[paramname: paramfunc]. For selecting params by string."""
    params_dict[func.__name__] = func
    return func


@dataclass
class Param:
    """Base class for all parameters.

    Parameters are callables that return parameter values.
    """

    def __call__(self) -> np.ndarray: ...


@_register_param
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

    def __str__(self):
        return f"{self.__class__.__name__}(value={self.value})"

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value})"


@_register_param
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

    def __str__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


@_register_param
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

    def __str__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper})"

    def __repr__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper})"


@dataclass
class Augmentation:
    """Base class for all augmentations.

    Augmentations are callables that return the augmented input.
    Can optionally pass through a second input.

    batch_x is expected to be [batch, time, channel].
    _apply works on [time, channel] inputs
    """

    def __call__(self, batch_x, batch_y=None):
        this_batch_x = batch_x.copy()
        for batch in range(batch_x.shape[0]):
            this_batch_x[batch, ...] = self._apply(this_batch_x[batch, ...])
        if batch_y is None:
            return this_batch_x
        else:
            return this_batch_x, batch_y

    def _apply(self, x: np.ndarray) -> np.ndarray:
        return x


@_register_augmentation
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

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(gain={self.gain.__repr__()})"


@_register_augmentation
class NormalizeMax(Augmentation):
    """Multiply signal with gain factor."""

    def __init__(self):
        pass

    def _apply(self, x):
        x /= np.nanmax(np.abs(x), initial=1.0)
        return x


@_register_augmentation
class NormalizePercentile(Augmentation):
    """Multiply signal with gain factor."""

    def __init__(self, percentile: Param):
        self.percentile = percentile

    def _apply(self, x):
        x /= np.nanpercentile(x, self.percentile())
        return x


@_register_augmentation
class NormalizeStd(Augmentation):
    """Multiply signal with gain factor."""

    def __init__(self):
        pass

    def _apply(self, x):
        x /= np.nanstd(x)
        return x


@_register_augmentation
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


@_register_augmentation
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


@_register_augmentation
class Upsampling(Augmentation):
    """Upsample signal."""

    def __init__(self, factor: Param):
        """
        Args:
            factor (Param): Upsampling factor.

        Raises:
            ValueError: if 'lower' attr of factor is <1 (would correspond to downsampling).
        """
        if hasattr(factor, "lower") and factor.lower < 1:  # type: ignore
            raise ValueError(f"Factor is {factor} - at the moment only upsampling (factors >= 1) is allowed.")
        self.factor = factor

    def _apply(self, x):
        len_x = x.shape[0]
        x = scipy.signal.resample_poly(x, up=int(len_x * self.factor()), down=len_x)
        x = x[:len_x]
        return x


@_register_augmentation
class MaskNoise(Augmentation):
    """Add noise or replace signal by noise for the full duration or a part of it."""

    def __init__(
        self, std: Optional[Param] = None, mean: Optional[Param] = None, duration: Optional[Param] = None, add: bool = True
    ):
        """
        Args:
            std (Optional[Param]): std of noise. Defaults to 1.
            mean (Optional[Param]): mean of noise. Defaults to 0.
            duration (Optional[Param]): nb_samples, Optional. If omitted will mask full duration.
            add (bool): add or replace. Defaults to True.
        """
        if std is None:
            std = Constant(1)
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
        noise = np.random.randn(duration, *x.shape[1:]) * self.std() + self.mean()
        if self.add:
            x[mask_start : mask_start + duration, :] += noise
        else:
            x[mask_start : mask_start + duration, :] = noise
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.mean}, mean={self.mean})"


@_register_augmentation
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
        duration = int(self.duration())
        mask_start = np.random.randint(low=0, high=len_x - duration)
        x[mask_start : mask_start + duration, :] = np.mean(x[mask_start : mask_start + duration], axis=0)
        return x


@_register_augmentation
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


@_register_augmentation
class NotchFilter(Augmentation):
    """Notch filter."""

    def __init__(self, freq: Param, Q: float = 30, samplerate_Hz: float = 10_000):
        """
        Args:
            freq (Param): freq to remove, in Hz.
        """
        self.freq = freq
        self.Q = Q
        self.samplerate_Hz = samplerate_Hz

    def _apply(self, x):
        b, a = scipy.signal.iirnotch(self.freq(), self.Q, self.samplerate_Hz)
        x = scipy.signal.filtfilt(b, a, x, axis=0)
        return x


class Augmentations:
    """Bundles several augmentations."""

    def __init__(self, augmentations: List[Augmentation]):
        """
        Args:
            augmentations (List[Augmentation]): List of Augmentation instances.
        """
        self.augmentations = augmentations

    def __call__(self, batch_x, batch_y=None):
        for augmentation in self.augmentations:
            batch_x = augmentation(batch_x)
        if batch_y is None:
            return batch_x
        else:
            return batch_x, batch_y

    def __len__(self):
        return len(self.augmentations)

    @classmethod
    def from_yaml(cls, filename: str):
        aug_spec = yaml.safe_load(open(filename, "r"))
        return cls.from_dict(aug_spec)

    @classmethod
    def from_dict(cls, aug_spec: Dict):
        augs = []
        for name, args in aug_spec.items():
            name = name.split("-", 1)[0]  # split off "-WHATEVER" suffix
            params = dict()
            if args is not None:  # for augs without args
                for a_name, a_arg in args.items():
                    if isinstance(a_arg, dict):  # Param-type arg
                        p_name = list(a_arg.keys())[0]
                        p_args = a_arg[p_name]
                        # if no args for Params are provided, use defaults
                        if p_args is None:
                            p_args = {}
                        params[a_name] = params_dict[p_name](**p_args)
                    else:  # standard arg
                        params[a_name] = a_arg

            logger.debug(params)
            aug = aug_dict[name](**params)
            logger.debug(aug)
            augs.append(aug)
        return cls(augs)
