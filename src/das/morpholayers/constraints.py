import tensorflow as tf
import numpy as np
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import skimage.morphology as skm
import scipy.ndimage.morphology as snm

MIN_LATT = -1
MAX_LATT = 0


@tf.custom_gradient
def rounding_op1(x):
    """Round the input tensor to the nearest tenth.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Rounded tensor.
    """

    def grad(dy):
        return dy

    return tf.round(x * 10) / 10, grad


@tf.custom_gradient
def rounding_op2(x):
    """Round the input tensor to the nearest hundredth.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Rounded tensor.
    """

    def grad(dy):
        return dy

    return tf.round(x * 100) / 100, grad


@tf.custom_gradient
def rounding_op3(x):
    """Round the input tensor to the nearest thousandth.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Rounded tensor.
    """

    def grad(dy):
        return dy

    return tf.round(x * 1000) / 1000, grad


@tf.custom_gradient
def rounding_op4(x):
    """Round the input tensor to the nearest ten-thousandth.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Rounded tensor.
    """

    def grad(dy):
        return dy

    return tf.round(x * 10000) / 10000, grad


class Rounding(Constraint):
    """Constrains weights by rounding them to specified decimal places.

    Args:
        c (int): Number of decimal places for rounding.
    """

    def __init__(self, c=4):
        self.c = c

    def __call__(self, w):
        if self.c == 1:
            return rounding_op1(w)
        elif self.c == 2:
            return rounding_op2(w)
        elif self.c == 3:
            return rounding_op3(w)
        else:
            # self.c==4:
            return rounding_op4(w)

    def get_config(self):
        return {"c": self.c}


class NonPositive(Constraint):
    """Constrains weights to be non-positive values."""

    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = MAX_LATT

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class NonPositiveExtensive(Constraint):
    """Constrains weights to be non-positive and centers equal to zero."""

    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = MAX_LATT

    def __call__(self, w):
        w = K.clip(w, self.min_value, 0)
        data = np.ones(w.shape)
        data[int(w.shape[0] / 2), int(w.shape[1] / 2), :, :] = 0
        w = tf.multiply(w, tf.convert_to_tensor(data, np.float32))
        return w

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class ZeroToOne(Constraint):
    """Constrains weights to be between 0 and 1."""

    def __init__(self):
        self.min_value = 0.0
        self.max_value = 1.0

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class Lattice(Constraint):
    """Constrains weights to be within a lattice range."""

    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT

    def __call__(self, w):
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class SEconstraint(Constraint):
    """Constrains weights to any structured element (SE) shape."""

    def __init__(self, SE=skm.disk(1)):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT
        self.data = SE

    def __call__(self, w):
        data = self.data
        data = np.repeat(data[:, :, np.newaxis], w.shape[2], axis=2)
        data = np.repeat(data[:, :, :, np.newaxis], w.shape[3], axis=3)
        w = w + (tf.convert_to_tensor(data, np.float32) + self.min_value)
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value, "SE": self.data}


class Disk(Constraint):
    """Constrains weights to a disk shape (only for square filters)."""

    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT

    def __call__(self, w):
        data = skm.disk(int(w.shape[0] / 2))
        data = np.repeat(data[:, :, np.newaxis], w.shape[2], axis=2)
        data = np.repeat(data[:, :, :, np.newaxis], w.shape[3], axis=3)
        w = w + (tf.convert_to_tensor(data, np.float32) + self.min_value)
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class Diamond(Constraint):
    """Constrains weights to a diamond shape (only for square filters)."""

    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT

    def __call__(self, w):
        data = skm.diamond(int(w.shape[0] / 2))
        data = np.repeat(data[:, :, np.newaxis], w.shape[2], axis=2)
        data = np.repeat(data[:, :, :, np.newaxis], w.shape[3], axis=3)
        w = w + (tf.convert_to_tensor(data, np.float32) + self.min_value)
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}
