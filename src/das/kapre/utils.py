# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from . import backend
from . import backend_keras
from typing import Optional


class AmplitudeToDB(Layer):
    """Converts spectrogram values to decibels.

    Examples:
        Adding dB conversion after a spectrogram:

        >>> model.add(Spectrogram(return_decibel=False))
        >>> model.add(AmplitudeToDB())
        which is the same as:
        >>> model.add(Spectrogram(return_decibel=True))

    """

    def __init__(self, amin: float = 1e-10, top_db: float = 80.0, **kwargs):
        """Args:
        amin (float, optional): Noise floor. Defaults to 1e-10 (dB).
        top_db (float, optional): Dynamic range of output. Defaults to 80.0 (dB).
        """
        self.amin = amin
        self.top_db = top_db
        super(AmplitudeToDB, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return backend_keras.amplitude_to_decibel(x, amin=self.amin, dynamic_range=self.top_db)

    def get_config(self):
        config = {"amin": self.amin, "top_db": self.top_db}
        base_config = super(AmplitudeToDB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Normalization2D(Layer):
    """Normalizes input along an axis.

    Examples:
        A frequency-axis normalization after a spectrogram:

        >>> model.add(Spectrogram())
        >>> model.add(Normalization2D(str_axis='freq'))

    """

    def __init__(
        self,
        str_axis: Optional[str] = None,
        int_axis: Optional[int] = None,
        image_data_format: str = "default",
        eps: float = 1e-10,
        **kwargs
    ):
        """[summary]

        Args:
            str_axis (Optional[str], optional):
                                    Axis name along which mean/std is computed (`batch`, `data_sample`, `channel`, `freq`, `time`).
                                    Recommended over `int_axis` because it provides more meaningful and image data format-robust interface.
                                    Defaults to None.
            int_axis (Optional[int], optional):
                                    Axis index along which mean/std is computed.
                                        - `0` for per data sample, `-1` for per batch.
                                        - `1`, `2`, `3` for channel, row, col (if channels_first)
                                    Defaults to None.
            image_data_format (str, optional): 'channels_first' (c,x,y,) or 'channels_last' (x,y,c) or TF 'default'. Defaults to 'default'.
            eps (float, optional): Small numerical value added to avoid divide by zero. Defaults to 1e-10.
        """
        assert not (int_axis is None and str_axis is None), "In Normalization2D, int_axis or str_axis should be specified."

        assert image_data_format in (
            "channels_first",
            "channels_last",
            "default",
        ), "Incorrect image_data_format: {}".format(image_data_format)

        if image_data_format == "default":
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.str_axis = str_axis
        if self.str_axis is None:  # use int_axis
            self.int_axis = int_axis
        else:  # use str_axis
            # warning
            if int_axis is not None:
                print("int_axis={} passed but is ignored, str_axis is used instead.".format(int_axis))
            # do the work
            assert str_axis in (
                "batch",
                "data_sample",
                "channel",
                "freq",
                "time",
            ), "Incorrect str_axis: {}".format(str_axis)
            if str_axis == "batch":
                int_axis = -1
            else:
                if self.image_data_format == "channels_first":
                    int_axis = ["data_sample", "channel", "freq", "time"].index(str_axis)
                else:
                    int_axis = ["data_sample", "freq", "time", "channel"].index(str_axis)

        assert int_axis in (-1, 0, 1, 2, 3), "invalid int_axis: " + str(int_axis)
        self.axis = int_axis
        self.eps = eps
        super(Normalization2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.axis == -1:
            mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = K.mean(x, axis=all_dims, keepdims=True)
            std = K.std(x, axis=all_dims, keepdims=True)
        return (x - mean) / (std + self.eps)

    def get_config(self):
        config = {
            "int_axis": self.axis,
            "str_axis": self.str_axis,
            "image_data_format": self.image_data_format,
        }
        base_config = super(Normalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
