# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from tensorflow.keras import backend as K


def amplitude_to_decibel(x, amin: float = 1e-10, dynamic_range: float = 80.0):
    """Convert (linear) amplitude to decibel (log10(x)).

        >>> x[x<amin] = amin  # clip everythin below amin
        >>> y = 10 * log(x) / log(10)  # log transform
        >>> y = ...  # rescale dyn range to [-80, 0]

    Args:
        x (Tensor): Tensor
        amin (float, optional): Minimal amplitude. Smaller values are clipped to this. Defaults to 1e-10 (dB).
        dynamic_range (float, optional): Dynamic range. Defaults to 80.0 (dB).

    Returns:
        [Tensor]: Tensor with the values converted to dB
    """
    log_spec = 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx())
    if K.ndim(x) > 1:
        axis = tuple(range(K.ndim(x))[1:])
    else:
        axis = None

    log_spec = log_spec - K.max(log_spec, axis=axis, keepdims=True)  # [-?, 0]
    log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
    return log_spec
