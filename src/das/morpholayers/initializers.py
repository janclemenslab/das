import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Initializer
import skimage.morphology as skm
import scipy.ndimage.morphology as snm

MIN_LATT = -1
MAX_LATT = 0


class MinusOnesZeroCenter(Initializer):
    """Initializer that generates tensors initialized to -1 except for center value.

    Args:
        None

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __call__(self, shape, dtype=None):
        data = -np.ones(shape)
        data[int(shape[0] / 2), int(shape[1] / 2), :, :] = 0
        return tf.convert_to_tensor(data, np.float32)


class SparseZeros(Initializer):
    """Initializer that generates tensors initialized to MIN_LATT except for center value.

    Args:
        th (float): Threshold for generating sparse zeros.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, th=0.85):
        self.th = th

    def __call__(self, shape, dtype=None):
        data = np.random.random(shape)
        data = (data > self.th) * 1.0
        data = data + MIN_LATT
        return tf.convert_to_tensor(data, np.float32)

    def get_config(self):
        return {"th": self.th}


class SparseNumZeros(Initializer):
    """Initializer that generates tensors initialized to MIN_LATT except for center value.

    Args:
        th (int): Number of zeros to generate.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, th=0):
        self.th = th

    def __call__(self, shape, dtype=None):
        data = np.random.random(shape)
        v = np.sort(data.flatten())
        data = (data <= v[self.th]) * 1.0
        data = data + MIN_LATT
        return tf.convert_to_tensor(data, np.float32)

    def get_config(self):
        return {"th": self.th}


class SignedOnes(Initializer):
    """Initializer that generates tensors initialized to random -1 or 1 values.

    Args:
        minval (float): Lower bound of the range of random values.
        maxval (float): Upper bound of the range of random values.
        seed (int): Seed for the random generator.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, minval=MIN_LATT, maxval=-MIN_LATT, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
        data = K.sign(K.random_uniform(shape, self.minval, self.maxval, dtype=dtype, seed=self.seed))
        if self.seed is not None:
            self.seed += 1
        return data

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class MinusOnes(Initializer):
    """Initializer that generates tensors initialized to -1.

    Args:
        None

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __call__(self, shape, dtype=None):
        return K.constant(MIN_LATT, shape=shape, dtype=dtype)


class RandomLattice(Initializer):
    """Initializer that generates tensors with a uniform distribution (MIN_LATT,MAX_LATT).

    Args:
        minval (float): Lower bound of the range of random values.
        maxval (float): Upper bound of the range of random values.
        seed (int): Seed for the random generator.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, minval=MIN_LATT, maxval=MAX_LATT, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
        data = K.random_uniform(shape, self.minval, self.maxval, dtype=dtype, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return data

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class RandomLatticewithZero(Initializer):
    """Initializer that generates tensors with a uniform distribution (MIN_LATT,-MIN_LATT).

    Args:
        minval (float): Lower bound of the range of random values.
        maxval (float): Upper bound of the range of random values.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, minval=MIN_LATT, maxval=MAX_LATT):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=None):
        data = K.random_uniform(shape, self.minval, self.maxval, dtype=dtype)
        data[int(shape[0] / 2), int(shape[1] / 2), :, :] = 0
        return tf.convert_to_tensor(data, np.float32)

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval}


class Quadratic(Initializer):
    """Initializer with quadratic values.

    Args:
        tvalue (float): T-value for distance transform.
        cvalue (float): Constant multiplier.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, tvalue=2, cvalue=0.2):
        self.tvalue = tvalue
        self.cvalue = cvalue

    def __call__(self, shape, dtype=None):
        data = np.ones([shape[0], shape[1]])
        data[int(shape[0] / 2), int(shape[1] / 2)] = 0
        data = (snm.distance_transform_edt(data) / self.tvalue) ** 2
        data = -self.cvalue * (data)
        data = np.repeat(data[:, :, np.newaxis], shape[2], axis=2)
        data = np.repeat(data[:, :, :, np.newaxis], shape[3], axis=3)
        return tf.convert_to_tensor(data, np.float32)

    def get_config(self):
        return {"t_value": self.t_value, "c_value": self.c_value}


class SEinitializer(Initializer):
    """Initializer to a Structured Element (SE).

    Args:
        SE (np.ndarray): Structured Element.
        minval (float): Minimum value.

    Returns:
        tf.Tensor: Initialized tensor.
    """

    def __init__(self, SE=None, minval=None):
        self.SE = SE
        if minval is None:
            self.minval = MIN_LATT
        else:
            self.minval = minval

    def __call__(self, shape, dtype=None):
        data = np.zeros(shape)
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                data[:, :, i, j] = self.SE + self.minval
        return tf.convert_to_tensor(data, np.float32)
