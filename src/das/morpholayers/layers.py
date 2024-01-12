"""
References
----------
Implementation of Morphological Layers:

Serra, J. (1983) Image Analysis and Mathematical Morphology.
       Academic Press, Inc. Orlando, FL, USA
Soille, P. (1999). Morphological Image Analysis. Springer-Verlag
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

K.set_image_data_format("channels_last")
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import InputSpec
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations
from .constraints import SEconstraint, ZeroToOne
from .initializers import MinusOnesZeroCenter
import skimage.morphology as skm
import scipy.ndimage.morphology as snm
from skimage.draw import line


def get_lines(sw):
    """Get lines.

    Args:
    sw (int): Size of the lines.

    Returns:
    tf.Tensor: Tensor containing lines.
    """
    FilterLines = []
    for i in range(sw):
        img = np.zeros((sw, sw), dtype=np.float32)
        rr, cc = line(i, 0, sw - i - 1, sw - 1)
        img[rr, cc] = 1
        FilterLines.append(img)
    for i in range(1, sw - 1):
        img = np.zeros((sw, sw), dtype=np.float32)
        rr, cc = line(0, i, sw - 1, sw - i - 1)
        img[rr, cc] = 1
        FilterLines.append(img)
    return K.stack(FilterLines, axis=-1)


"""Classical Operators"""


@tf.function
def convolution2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Convolution Operator (Depthwise).

    Args:
    x : tf.Tensor
        Input tensor.
    st_element : tf.Tensor
        Nonflat structuring element.
    strides : tuple
        Strides as classical convolutional layers.
    padding : str
        Padding as classical convolutional layers.
    rates : tuple, optional
        Rates as classical convolutional layers.

    Returns
        tf.Tensor: Result of the convolution.

    """
    x = tf.nn.depthwise_conv2d(x, tf.expand_dims(st_element, 3), (1,) + strides + (1,), padding.upper(), "NHWC", rates)
    return x


@tf.function
def dilation2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Dilation Operator.

    Args:
    x : tf.Tensor
        Input tensor.
    st_element : tf.Tensor
        Nonflat structuring element.
    strides : tuple
        Strides as classical convolutional layers.
    padding : str
        Padding as classical convolutional layers.
    rates : tuple, optional
        Rates as classical convolutional layers.

    Returns
    -------
    tf.Tensor
        Result of the dilation.

    """
    x = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


@tf.function
def erosion2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Erosion Operator.

    Args:
    x : tf.Tensor
        Input tensor.
    st_element : tf.Tensor
        Nonflat structuring element.
    strides : tuple
        Strides as classical convolutional layers.
    padding : str
        Padding as classical convolutional layers.
    rates : tuple, optional
        Rates as classical convolutional layers.

    Returns
    -------
    tf.Tensor
        Result of the erosion.

    """
    x = tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


@tf.function
def opening2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Opening Operator.

    Args:
    x : tf.Tensor
        Input tensor.
    st_element : tf.Tensor
        Nonflat structuring element.
    strides : tuple
        Strides are only applied in the second operator (dilation).
    padding : str
        Padding as classical convolutional layers.
    rates : tuple, optional
        Rates are only applied in the second operator (dilation).

    Returns
    -------
    tf.Tensor
        Result of the opening operation.

    """
    x = tf.nn.erosion2d(x, st_element, (1,) + (1, 1) + (1,), padding.upper(), "NHWC", (1,) + (1, 1) + (1,))
    x = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


@tf.function
def closing2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Closing Operator.

    Args:
    x : tf.Tensor
        Input tensor.
    st_element : tf.Tensor
        Nonflat structuring element.
    strides : tuple
        Strides are only applied in the second operator (erosion).
    padding : str
        Padding as classical convolutional layers.
    rates : tuple, optional
        Rates are only applied in the second operator (erosion).

    Returns
    -------
    tf.Tensor
        Result of the closing operation.

    """
    x = tf.nn.dilation2d(x, st_element, (1,) + (1, 1) + (1,), padding.upper(), "NHWC", (1,) + (1, 1) + (1,))
    x = tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


@tf.function
def gradient2d(x, st_element, strides, padding, rates=(1, 1)):
    """Gradient Operator.

    Args:
        x : tf.Tensor Input tensor.
        st_element (tf.Tensor): Nonflat structuring element.
        strides (tuple): Strides are only applied in the second operator (erosion).
        padding (str): Padding as classical convolutional layers.
        rates (tuple, optional): Rates are only applied in the second operator (erosion).

    Returns
        tf.Tensor: Result of the gradient operation.
    """
    x = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,)) - tf.nn.erosion2d(
        x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,)
    )
    return x


@tf.function
def internalgradient2d(x, st_element, strides, padding, rates=(1, 1)):
    """Internal Gradient Operator

    Args:
        x: Input tensor.
        st_element: Nonflat structuring element.
        strides: Strides applied in the second operator (erosion).
        padding: Padding as classical convolutional layers.
        rates: Rates applied in the second operator (erosion).

    Returns:
        Tensor after applying the internal gradient operator.
    """
    x = x - tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


@tf.function
def externalgradient2d(x, st_element, strides, padding, rates=(1, 1)):
    """External Gradient Operator

    Args:
        x: Input tensor.
        st_element: Nonflat structuring element.
        strides: Strides applied in the second operator (erosion).
        padding: Padding as classical convolutional layers.
        rates: Rates applied in the second operator (erosion).

    Returns:
        Tensor after applying the external gradient operator.
    """
    x = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,)) - x
    return x


@tf.function
def togglemapping2d(x, st_element, strides=(1, 1), padding="same", rates=(1, 1), steps=5):
    """Toggle Mapping Operator

    Args:
        x: Input tensor.
        st_element: Nonflat structuring element.
        strides: Strides applied in the second operator (erosion).
        padding: Padding as classical convolutional layers.
        rates: Rates applied in the second operator (erosion).
        steps: Number of toggle mapping steps.

    Returns:
        Tensor after applying the toggle mapping operator.
    """
    for _ in range(steps):
        d = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        e = tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        Delta = tf.keras.layers.Minimum()([tf.abs(d - x), tf.abs(x - e)])
        Mask = tf.cast(tf.less_equal(d - x, x - e), "float32")
        x = x + (Mask * Delta)
    return x


@tf.function
def togglemapping(X, steps=5):
    """K steps of toggle mapping operator

    Args:
        X: Input tensor.
        steps: Number of toggle mapping steps.

    Returns:
        Tensor after applying K steps of toggle mapping operator.
    """
    for _ in range(steps):
        d = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(X)
        e = -tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(-X)  # MinPooling
        Delta = tf.keras.layers.Minimum()([d - X, X - e])
        Mask = tf.cast(tf.less_equal(d - X, X - e), "float32")
        X = X + (Mask * Delta)
    return X


@tf.function
def antidilation2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Dilation Operator of the negative of the input image

    Args:
        x: Input tensor.
        st_element: Nonflat structuring element.
        strides: Strides as classical convolutional layers.
        padding: Padding as classical convolutional layers.
        rates: Rates as classical convolutional layers.

    Returns:
        Tensor after applying the basic dilation operator.
    """
    x = tf.nn.dilation2d(-x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


@tf.function
def antierosion2d(x, st_element, strides, padding, rates=(1, 1)):
    """Basic Erosion Operator of the negative of the input image

    Args:
        x: Input tensor.
        st_element: Nonflat structuring element.
        strides: Strides as classical convolutional layers.
        padding: Padding as classical convolutional layers.
        rates: Rates as classical convolutional layers.

    Returns:
        Tensor after applying the basic erosion operator.
    """
    x = tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight("bias", shape=input_shape[1:], initializer="zeros", trainable=True)

    def call(self, x):
        return x + self.bias


"""Operator by Reconstruction"""


@tf.function
def condition_equal(last, new, image):
    """Check if two tensors are not equal element-wise.

    Args:
        last: Previous tensor.
        new: Current tensor.
        image: Image tensor.

    Returns:
        Boolean tensor indicating if the tensors are not equal element-wise.
    """
    return tf.math.logical_not(tf.reduce_all(tf.math.equal(last, new)))


def update_dilation(last, new, mask):
    """Update the dilation step during reconstruction.

    Args:
        last: Previous tensor.
        new: Current tensor.
        mask: Mask tensor.

    Returns:
        Updated tensors for the next dilation step.
    """
    return [new, geodesic_dilation_step([new, mask]), mask]


@tf.function
def geodesic_dilation_step(X):
    """1 step of reconstruction by dilation.

    Args:
        X: Input tensor.

    Returns:
        Tensor after one step of geodesic dilation.
    """
    return tf.keras.layers.Minimum()(
        [tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(X[0]), X[1]]
    )


@tf.function
def geodesic_dilation(X, steps=None):
    """Full reconstruction by dilation if steps=None, else
    K steps reconstruction by dilation.

    Args:
        X: Input tensor.
        steps: Number of steps (None for complete reconstruction).

    Returns:
        Tensor after geodesic dilation.
    """
    rec = X[0]
    rec = geodesic_dilation_step([rec, X[1]])
    _, rec, _ = tf.while_loop(condition_equal, update_dilation, [X[0], rec, X[1]], maximum_iterations=steps)
    return rec


def reconstruction_dilation(X):
    """Full geodesic reconstruction by dilation, reaching idempotence.

    Args:
        X: Input tensor.

    Returns:
        Tensor after full geodesic reconstruction by dilation.
    """
    return geodesic_dilation(X, steps=None)


def update_erosion(last, new, mask):
    """Update the erosion step during reconstruction.

    Args:
        last: Previous tensor.
        new: Current tensor.
        mask: Mask tensor.

    Returns:
        Updated tensors for the next erosion step.
    """
    return [new, geodesic_erosion_step([new, mask]), mask]


@tf.function
def geodesic_erosion_step(X):
    """
    1 step of reconstruction by erosion.

    Args:
        X: Input tensor.

    Returns:
        Tensor after one step of geodesic erosion.
    """
    return tf.keras.layers.Maximum()(
        [-tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(-X[0]), X[1]]
    )


@tf.function
def geodesic_erosion(X, steps=None):
    """Full reconstruction by erosion if steps=None, else
    K steps reconstruction by erosion.

    Args:
        X: Input tensor.
        steps: Number of steps (None for complete reconstruction).

    Returns:
        Tensor after geodesic erosion.
    """
    rec = X[0]
    rec = geodesic_erosion_step([rec, X[1]])
    _, rec, _ = tf.while_loop(condition_equal, update_erosion, [X[0], rec, X[1]], maximum_iterations=steps)

    return rec


def reconstruction_erosion(X, steps=None):
    """Full geodesic reconstruction by erosion, reaching idempotence

    Args:
        X: Input tensor.
        steps: Number of steps (by default None).

    Returns:
        Tensor after full geodesic reconstruction by erosion.
    """
    return geodesic_erosion(X, steps=None)


@tf.function
def leveling_iteration(X):
    """K steps of reconstruction by dilation

    Args:
        X: Input tensor.

    Returns:
        Tensor after K steps of reconstruction by dilation.
    """
    return tf.keras.layers.Maximum()(
        [
            -tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(-X[0]),
            tf.keras.layers.Minimum()(
                [tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(X[0]), X[1]]
            ),
        ]
    )


def update_leveling(last, new, mask):
    return new, leveling_iteration([new, mask]), mask


@tf.function
def condition_nonzero(new, count):
    return tf.math.logical_not(tf.reduce_all(tf.math.not_equal(count, 0.0)))


@tf.function
def update_distance(new, count):
    return distance_step([new, count])


@tf.function
def distance_step(X):
    """One step of morphological distance by dilation

    Args:
        X: Input tensor.

    Returns:
        Tensor after one step of morphological distance by dilation.
    """
    Z = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(X[0])
    return [Z, Z + X[1]]


@tf.function
def morphological_distance(X, steps=None):
    """Morphological Distance Transform if steps=None, else
    K steps morphological Distance

    Args:
        X: Input tensor.
        steps: Number of steps (None means iterate until non-zero values).

    Returns:
        Tensor after morphological distance transform.
    """
    count = X
    new = X
    _, D = tf.while_loop(condition_nonzero, update_distance, [new, count], maximum_iterations=steps)
    return tf.math.reduce_max(D, keepdims=True, axis=[1, 2, 3]) - D


@tf.function
def leveling(X, steps=None):
    """Perform Leveling from Marker

    Args:
        X: Input tensor.
        steps: Number of steps.

    Returns:
        Tensor after performing leveling from marker.
    """
    lev = leveling_iteration([X[0], X[1]])
    _, lev, _ = tf.while_loop(condition_equal, update_leveling, [X[0], lev, X[1]], maximum_iterations=steps)
    return lev


"""Reconstruction based operators"""


@tf.function
def h_maxima_transform(X):
    """H-maxima transform of image X[1] with h=X[0]

    Args:
        X: Input tensor.

    Returns:
        Tensor after h-maxima transform.
    """
    h = X[0]
    Mask = X[1]
    HMAX = geodesic_dilation([Mask - h, Mask])
    return HMAX


@tf.function
def h_minima_transform(X):
    """H-maxima transform of image X[1] with h=X[0]

    Args:
        X: Input tensor.

    Returns:
        Tensor after h-minima transform.
    """
    h = X[0]
    Mask = X[1]

    HMIN = geodesic_erosion([Mask + h, Mask])
    return HMIN


@tf.function
def h_convex_transform(X):
    """H-convex transform of image X[1] with h=X[0]

    Args:
        X: Input tensor.

    Returns:
        Tensor after h-convex transform.
    """
    h = X[0]
    Mask = X[1]
    HCONVEX = Mask - geodesic_dilation([Mask - h, Mask])
    return HCONVEX


@tf.function
def h_concave_transform(X):
    """H-convex transform of image X[1] with h=X[0]

    Args:
        X: Input tensor.

    Returns:
        Tensor after h-concave transform.
    """
    h = X[0]
    Mask = X[1]
    HCONCAVE = geodesic_erosion([Mask + h, Mask]) - Mask
    return HCONCAVE


@tf.function
def region_maxima_transform(X):
    """Region maxima transform of image X

    Args:
        X: Input tensor.

    Returns:
        Tensor after region maxima transform.
    """
    return h_convex_transform([tf.convert_to_tensor([[1.0 / 255.0]]), X])


@tf.function
def region_minima_transform(X):
    """Region minima transform of image X

    Args:
        X: Input tensor.

    Returns:
        Tensor after region minima transform.
    """
    return h_concave_transform([tf.convert_to_tensor([[1.0 / 255.0]]), X])


@tf.function
def extended_maxima_transform(X):
    """Extended maxima transform of image X[1] with h=X[0]

    Args:
        X: Input tensor.

    Returns:
        Tensor after extended maxima transform.
    """
    return region_maxima_transform(h_maxima_transform(X))


@tf.function
def extended_minima_transform(X):
    """
    extended minima transform of image X[1] with h=X[0]

    Args:
        X: Input tensor.

    Returns:
        Tensor after extended minima transform.
    """
    return region_minima_transform(h_minima_transform(X))


"""
====================
Max/Min of Operators
====================
"""


class Erosion2D(Layer):
    """
    Sum of Depthwise (Marginal) Erosion 2D on the third axes
    for now assuming channel last

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    :Example:

    >>>from keras.models import Sequential,Model
    >>>from keras.layers import Input
    >>>xin=Input(shape=(28,28,3))
    >>>x=Erosion2D(num_filters=7,kernel_size=(5,5)))(xin)
    >>>model = Model(xin,x)

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(Erosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        # Be sure to call this at the end
        super(Erosion2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking max over channel_axis
            res.append(tf.reduce_sum(erosion2d(x, self.kernel[..., i], self.strides, self.padding), axis=-1))
        output = tf.stack(res, axis=-1)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class Dilation2D(Layer):
    """
    Sum of Depthwise (Marginal) Dilation 2D on the third axes
    for now assuming channel last

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    :Example:

    >>>from keras.models import Sequential,Model
    >>>from keras.layers import Input
    >>>xin=Input(shape=(28,28,3))
    >>>x=Dilation2D(num_filters=7,kernel_size=(5,5)))(xin)
    >>>model = Model(xin,x)

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(Dilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        # Be sure to call this at the end
        super(Dilation2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking max over channel_axis
            res.append(tf.reduce_sum(dilation2d(x, self.kernel[..., i], self.strides, self.padding), axis=-1))
        output = tf.stack(res, axis=-1)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class IntegratorofOperator2D(Layer):
    """
    Integrator on channel axis of 2D operator apply per channel
    for now assuming channel last

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    :Example:

    >>>from keras.models import Sequential,Model
    >>>from keras.layers import Input
    >>>xin=Input(shape=(28,28,3))
    >>>x=IntegratorofOperator2D(num_filters=7,kernel_size=(5,5),operator=dilation2d,integrator=K.max))(xin)
    >>>model = Model(xin,x)

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        integrator=K.sum,
        operator=dilation2d,
        **kwargs,
    ):
        super(IntegratorofOperator2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.integrator = integrator
        self.operator = operator

        # for we are assuming channel last
        self.channel_axis = -1
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        super(IntegratorofOperator2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            res.append(
                self.integrator(self.operator(x, self.kernel[..., i], self.strides, self.padding), axis=self.channel_axis)
            )
        output = tf.stack(res, axis=-1)
        # print('output.shape',output.shape)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)
        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
                "integrator": self.integrator,
                "operator": self.operator,
            }
        )
        return config


class MaxofErosions2D(Layer):
    """
    Maximum of Erosion 2D Layer
    for now assuming channel last
    [1]_, [2]_, [3]_, [4]_

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    :Example:

    >>>from keras.models import Sequential,Model
    >>>from keras.layers import Input
    >>>xin=Input(shape=(28,28,3))
    >>>x=MaxofErosion2D(num_filters=7,kernel_size=(5,5)))(xin)
    >>>model = Model(xin,x)

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(MaxofErosions2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(MaxofErosions2D, self).build(input_shape)

    def call(self, x):
        outputs = K.placeholder()
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking min over channel_axis
            out = erosion2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        outputs = K.max(outputs, axis=-1, keepdims=True)

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class MaxofErosions2D_old(Layer):
    """
    Maximum of Erosion 2D Layer
    for now assuming channel last
    [1]_, [2]_, [3]_, [4]_

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    :Example:

    >>>from keras.models import Sequential,Model
    >>>from keras.layers import Input
    >>>xin=Input(shape=(28,28,3))
    >>>x=MaxofErosion2D(num_filters=7,kernel_size=(5,5)))(xin)
    >>>model = Model(xin,x)

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(MaxofErosions2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(MaxofErosions2D, self).build(input_shape)

    def call(self, x):
        outputs = K.placeholder()
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking min over channel_axis
            out = K.max(erosion2d(x, self.kernel[..., i], self.strides, self.padding), axis=self.channel_axis, keepdims=True)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class MinofDilations2D(Layer):
    """
    Minimum of Dilations 2D Layer assuming channel last

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(MinofDilations2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(MinofDilations2D, self).build(input_shape)

    def call(self, x):
        # outputs = K.placeholder()
        for i in range(self.num_filters):
            # dilation2d returns image of same size as x
            # so taking max over channel_axis
            out = dilation2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        outputs = K.min(outputs, axis=-1, keepdims=True)

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class MinofDilations2D_old(Layer):
    """
    Minimum of Dilations 2D Layer assuming channel last

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(MinofDilations2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(MinofDilations2D, self).build(input_shape)

    def call(self, x):
        # outputs = K.placeholder()
        for i in range(self.num_filters):
            # dilation2d returns image of same size as x
            # so taking max over channel_axis
            out = K.min(dilation2d(x, self.kernel[..., i], self.strides, self.padding), axis=self.channel_axis, keepdims=True)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
===========================
Depthwise Layers (Marginal)
===========================
"""


class DepthwiseOperator2D(Layer):
    """
    Depthwise Operator 2D Layer: Depthwise Operator for now assuming channel last
    """

    def __init__(
        self,
        kernel_size,
        depth_multiplier=1,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        shared=False,
        operator=dilation2d,
        **kwargs,
    ):
        super(DepthwiseOperator2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared
        self.operator = operator
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        if self.shared:
            kernel_shape = self.kernel_size + (self.depth_multiplier,)
        else:
            kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier)
        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        super(DepthwiseOperator2D, self).build(input_shape)

    def call(self, x):
        res = []
        if self.shared:
            for di in range(self.depth_multiplier):
                H = self.operator(
                    x,
                    tf.repeat(self.kernel2D[:, :, di : (di + 1)], x.shape[-1], axis=2),
                    strides=self.strides,
                    padding=self.padding.upper(),
                    rates=self.rates,
                )
                res.append(H)
        else:
            for di in range(self.depth_multiplier):
                H = self.operator(
                    x, self.kernel2D[:, :, :, di], strides=self.strides, padding=self.padding.upper(), rates=self.rates
                )
                res.append(H)
        output = tf.concat(res, axis=-1)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class DepthwiseDilation2D(Layer):
    """
    Depthwise Dilation 2D Layer: Depthwise Dilation for now assuming channel last
    """

    def __init__(
        self,
        kernel_size,
        depth_multiplier=1,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        shared=False,
        **kwargs,
    ):
        super(DepthwiseDilation2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        if self.shared:
            kernel_shape = self.kernel_size + (self.depth_multiplier,)
        else:
            kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier)
        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        super(DepthwiseDilation2D, self).build(input_shape)

    def call(self, x):
        res = []
        if self.shared:
            for di in range(self.depth_multiplier):
                H = tf.nn.dilation2d(
                    x,
                    tf.repeat(self.kernel2D[:, :, di : (di + 1)], x.shape[-1], axis=2),
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                res.append(H)
        else:
            for di in range(self.depth_multiplier):
                H = tf.nn.dilation2d(
                    x,
                    self.kernel2D[:, :, :, di],
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                res.append(H)
        return tf.concat(res, axis=-1)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class DepthwiseErosion2D(Layer):
    """
    Depthwise Erosion 2D Layer: Depthwise Dilation for now assuming channel last
    """

    def __init__(
        self,
        kernel_size,
        depth_multiplier=1,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        shared=False,
        **kwargs,
    ):
        super(DepthwiseErosion2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]

        if self.shared:
            kernel_shape = self.kernel_size + (self.depth_multiplier,)
        else:
            kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier)
        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        super(DepthwiseErosion2D, self).build(input_shape)

    def call(self, x):
        res = []
        if self.shared:
            for di in range(self.depth_multiplier):
                H = tf.nn.erosion2d(
                    x,
                    tf.repeat(self.kernel2D[:, :, di : (di + 1)], x.shape[-1], axis=2),
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                res.append(H)
        else:
            for di in range(self.depth_multiplier):
                H = tf.nn.erosion2d(
                    x,
                    self.kernel2D[:, :, :, di],
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                res.append(H)
        return tf.concat(res, axis=-1)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
==============================
Separable Morphological Layers
==============================
"""


# TODO CHECK SEPARABLE (Not coherent with definition)
class SeparableDilation2D(Layer):
    """
    Separable Dilation 2D Layer: First Depthwise Dilation follows by a Pointwise Dilation
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        depth_multiplier=1,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        shared=False,
        **kwargs,
    ):
        super(SeparableDilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        if self.shared:
            kernel_shape = self.kernel_size + (self.num_filters,)
        else:
            kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        kernelDepth_shape = (input_dim, self.depth_multiplier)

        self.kernelDepth = self.add_weight(
            shape=kernelDepth_shape,
            initializer=self.kernel_initializer,
            name="kernelDepth",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        super(SeparableDilation2D, self).build(input_shape)

    def call(self, x):
        res = []
        if self.shared:
            for ki in range(self.num_filters):
                H = tf.nn.dilation2d(
                    x,
                    tf.repeat(self.kernel2D[:, :, ki : (ki + 1)], x.shape[-1], axis=2),
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                for i in range(self.depth_multiplier):
                    # Pointwise Max-Plus Convolution
                    res.append(tf.reduce_max(self.kernelDepth[..., i] + H, axis=-1))
        else:
            for ki in range(self.num_filters):
                H = tf.nn.dilation2d(
                    x,
                    self.kernel2D[:, :, :, ki],
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                for i in range(self.depth_multiplier):
                    # Pointwise Max-Plus Convolution
                    res.append(tf.reduce_max(self.kernelDepth[..., i] + H, axis=-1))
        return tf.stack(res, axis=-1)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class SeparableErosion2D(Layer):
    """
    Separable Erosion 2D Layer: First Depthwise Erosion follows by a Pointwise Erosion
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        depth_multiplier=1,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        shared=False,
        **kwargs,
    ):
        super(SeparableErosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        if self.shared:
            kernel_shape = self.kernel_size + (self.num_filters,)
        else:
            kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        kernelDepth_shape = (input_dim, self.depth_multiplier)

        self.kernelDepth = self.add_weight(
            shape=kernelDepth_shape,
            initializer=self.kernel_initializer,
            name="kernelDepth",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        super(SeparableErosion2D, self).build(input_shape)

    def call(self, x):
        res = []
        if self.shared:
            for ki in range(self.num_filters):
                H = tf.nn.erosion2d(
                    x,
                    tf.repeat(self.kernel2D[:, :, ki : (ki + 1)], x.shape[-1], axis=2),
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                for i in range(self.depth_multiplier):
                    # Pointwise Max-Plus Convolution
                    res.append(tf.reduce_max(self.kernelDepth[..., i] + H, axis=-1))
        else:
            for ki in range(self.num_filters):
                H = tf.nn.erosion2d(
                    x,
                    self.kernel2D[:, :, :, ki],
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format="NHWC",
                    dilations=(1,) + self.rates + (1,),
                )
                for i in range(self.depth_multiplier):
                    # Pointwise Max-Plus Convolution
                    res.append(tf.reduce_max(self.kernelDepth[..., i] + H, axis=-1))
        return tf.stack(res, axis=-1)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class BSErosion2D(Layer):
    """
    Blueprint Separable Erosion 2D Layer: First Pointwise follows by Depthwise Erosion
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        shared=False,
        **kwargs,
    ):
        super(BSErosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]

        if self.shared:
            kernel_shape = self.kernel_size + (1,)
        else:
            kernel_shape = self.kernel_size + (self.num_filters,)

        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        kernelDepth_shape = (input_dim, self.num_filters)

        self.kernelDepth = self.add_weight(
            shape=kernelDepth_shape,
            initializer=self.kernel_initializer,
            name="kernelDepth",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        super(BSErosion2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            res.append(tf.reduce_min(x - self.kernelDepth[..., i], axis=-1))
        res = tf.stack(res, axis=-1)
        if self.shared:
            return tf.nn.erosion2d(
                res,
                tf.repeat(self.kernel2D[:, :, 0:1], res.shape[-1], axis=2),
                strides=(1,) + self.strides + (1,),
                padding=self.padding.upper(),
                data_format="NHWC",
                dilations=(1,) + self.rates + (1,),
            )
        else:
            return tf.nn.erosion2d(
                res,
                self.kernel2D,
                strides=(1,) + self.strides + (1,),
                padding=self.padding.upper(),
                data_format="NHWC",
                dilations=(1,) + self.rates + (1,),
            )

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class BSDilation2D(Layer):
    """
    Blueprint Separable Dilation 2D Layer: First Pointwise follows by Depthwise Dilation
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        shared=False,
        **kwargs,
    ):
        super(BSDilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1
        self.shared = shared

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]

        if self.shared:
            kernel_shape = self.kernel_size + (1,)
        else:
            kernel_shape = self.kernel_size + (self.num_filters,)

        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        kernelDepth_shape = (input_dim, self.num_filters)

        self.kernelDepth = self.add_weight(
            shape=kernelDepth_shape,
            initializer=self.kernel_initializer,
            name="kernelDepth",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        super(BSDilation2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            # Pointwise Max-Plus Convolution
            res.append(tf.reduce_max(x + self.kernelDepth[..., i], axis=-1))
        res = tf.stack(res, axis=-1)
        if self.shared:
            return tf.nn.dilation2d(
                res,
                tf.repeat(self.kernel2D[:, :, 0:1], res.shape[-1], axis=2),
                strides=(1,) + self.strides + (1,),
                padding=self.padding.upper(),
                data_format="NHWC",
                dilations=(1,) + self.rates + (1,),
            )
        else:
            return tf.nn.dilation2d(
                res,
                self.kernel2D,
                strides=(1,) + self.strides + (1,),
                padding=self.padding.upper(),
                data_format="NHWC",
                dilations=(1,) + self.rates + (1,),
            )

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class SeparableDilation3D(Layer):
    """
    Separable Dilation 3D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        kernel_size,
        depth_multiplier=1,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(SeparableDilation3D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        # input_dim = input_shape[self.channel_axis]
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 1)

        self.kernel2D = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel2D",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )
        kernelDepth_shape = (self.kernel_size[2], 1, 1)

        self.kernelDepth = self.add_weight(
            shape=kernelDepth_shape,
            initializer=self.kernel_initializer,
            name="kernelDepth",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(SeparableDilation3D, self).build(input_shape)

    def call(self, x):
        H = tf.nn.dilation2d(
            x,
            tf.repeat(self.kernel2D, x.shape[-1], axis=2),
            strides=(1,) + self.strides + (1,),
            padding="SAME",
            data_format="NHWC",
            dilations=(1,) + self.rates + (1,),
        )
        H = tf.einsum("ijkl->iljk", H)
        H = tf.nn.dilation2d(
            H,
            tf.repeat(self.kernelDepth, H.shape[-1], axis=2),
            strides=(1, 1, 1, 1),
            padding="SAME",
            data_format="NHWC",
            dilations=(1, 1, 1, 1),
        )
        H = tf.einsum("ijkl->iklj", H)
        return H

        # res=[]
        # for ki in range(self.num_filters):
        #    H=tf.nn.dilation2d(x,tf.repeat(self.kernel2D[:,:,ki:(ki+1)],x.shape[-1],axis=2),strides=(1, ) + self.strides + (1, ),padding=self.padding.upper(),data_format="NHWC",dilations=(1,)+self.rates+(1,))
        #    for i in range(self.depth_multiplier):
        #        res.append(tf.reduce_max(self.kernelDepth[...,i]+H,axis=-1))
        #        #res.append(tf.reduce_max(H,axis=-1))
        # return tf.stack(res,axis=-1)

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (input_shape[-1],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
                "integrator": self.integrator,
                "operator": self.operator,
            }
        )
        return config


# TODO: CONVERT TO KERNEL_CONSTRAINT?
class DilationSE2D(Layer):
    """
    Dilation SE 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        structuring_element=skm.disk(1),
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(DilationSE2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = structuring_element.shape
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        # self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_constraint = SEconstraint(SE=structuring_element)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        super(DilationSE2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking max over channel_axis
            res.append(tf.reduce_sum(dilation2d(x, self.kernel[..., i], self.strides, self.padding), axis=-1))
        output = tf.stack(res, axis=-1)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class Antierosion2D(Layer):
    """
    AntiErosion 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(Antierosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(Antierosion2D, self).build(input_shape)

    def call(self, x):
        # outputs = K.placeholder()
        for i in range(self.num_filters):
            out = antierosion2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class Antidilation2D(Layer):
    """
    Antidilation 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(Antidilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(Antidilation2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = antidilation2d(x, self.kernel[..., i], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
=================================
Quadratic Morphological Operators
=================================
"""


class QuadraticDilation2D(Layer):
    """
    Quadratic Dilation 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        bias_initializer=tf.keras.initializers.RandomUniform(minval=0.1, maxval=1),
        bias_constraint=None,
        bias_regularization=None,
        scale=4,
        **kwargs,
    ):
        super(QuadraticDilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.scale = scale
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.bias_regularization = tf.keras.regularizers.get(bias_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]

        data = np.ones(self.kernel_size)
        data[int(data.shape[0] / 2), int(data.shape[1] / 2)] = 0
        # data=snm.distance_transform_edt(data)**2
        data = snm.distance_transform_edt(data)
        data = -((data / (2 * self.scale)) ** 2)  # TO CHECK EXPRESSIOn
        data = np.repeat(data[:, :, np.newaxis], input_dim, axis=2)
        data = np.repeat(data[:, :, :, np.newaxis], self.num_filters, axis=3)
        self.data = tf.convert_to_tensor(data, np.float32)
        self.bias = self.add_weight(
            shape=(input_dim, self.num_filters),
            initializer=self.bias_initializer,
            name="bias",
            constraint=self.bias_constraint,
            regularizer=self.bias_regularization,
        )
        super(QuadraticDilation2D, self).build(input_shape)

    def call(self, x):
        kernel = tf.math.multiply(self.data, self.bias)
        for i in range(self.num_filters):
            out = dilation2d(x, kernel[..., i], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
=================
Top Hat Operators
=================
"""


class TopHatOpening2D(Layer):
    """
    TopHat from Opening 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(TopHatOpening2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(TopHatOpening2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = self.tophatopening2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def tophatopening2d(self, x, st_element, strides, padding, rates=(1, 1)):
        z = tf.nn.erosion2d(x, st_element, (1,) + (1, 1) + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        z = tf.nn.dilation2d(z, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        return x - z

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class TopHatClosing2D(Layer):
    """
    TopHat from Closing 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(TopHatClosing2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1
        # TODO strides not working.

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(TopHatClosing2D, self).build(input_shape)

    def call(self, x):
        # outputs = K.placeholder()
        for i in range(self.num_filters):
            out = self.tophatclosing2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def tophatclosing2d(self, x, st_element, strides, padding, rates=(1, 1)):
        z = tf.nn.dilation2d(x, st_element, (1,) + (1, 1) + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        z = tf.nn.erosion2d(z, st_element, (1,) + (1, 1) + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        return z - x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
===================
Opening and Closing
===================
"""


class Opening2D(Layer):
    """
    Opening 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(Opening2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(Opening2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = opening2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class Closing2D(Layer):
    """
    Closing 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(Closing2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(Closing2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = closing2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=1
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
============================
Special Structuring Elements
============================
"""


class DepthwiseDilationLines2D(Layer):
    """
    Depthwise Dilation Lines 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(DepthwiseDilationLines2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # self.mask = tensorflow.Variable(initial_value=get_lines(self.kernel_size[0],input_dim=input_shape[-1]),trainable=False)
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # print('Dimension Input',input_shape)

    def build(self, input_shape):
        # print('Dimension Input',input_shape)
        self.mask = tf.Variable(initial_value=get_lines(self.kernel_size[0]), trainable=False)
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        # input_dim = input_shape[self.channel_axis]
        self.kernel = self.add_weight(
            shape=self.mask.shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.kernel.shape[-1] * input_shape[self.channel_axis],),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        super(DepthwiseDilationLines2D, self).build(input_shape)

    def call(self, x):
        # print('self.kernel.shape',self.kernel.shape)
        # print('x.shape',x.shape)
        for i in range(x.shape[-1]):
            for j in range(self.kernel.shape[-1]):
                out = dilation2d(
                    x[..., i : (i + 1)],
                    self.kernel[..., j : (j + 1)] * self.mask[..., j : (j + 1)],
                    self.strides,
                    self.padding,
                    self.rates,
                )
                if (i == 0) and (j == 0):
                    outputs = out
                else:
                    outputs = K.concatenate([outputs, out])
        if self.use_bias:
            outputs = tf.keras.backend.bias_add(outputs, self.bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

        # res=[]
        # for i in range(self.num_filters):
        # erosion2d returns image of same size as x
        # so taking max over channel_axis
        #    res.append(tf.reduce_sum(dilation2d(x, self.kernel[..., i],self.strides, self.padding),axis=-1))
        # output= tf.stack(res,axis=-1)
        # return output
        # if self.use_bias:
        #     output=tf.keras.backend.bias_add(output, self.bias)

        # if self.activation is not None:
        #     return self.activation(output)
        # return output

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.kernel.shape[-1] * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
                "mask": self.mask,
            }
        )
        return config


"""
==============
Morpho Average
==============
"""


class MorphoAverage2D(Layer):
    """
    Average of Dilation and Erosion2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(MorphoAverage2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        # Be sure to call this at the end
        super(MorphoAverage2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = self.emd2d(x, self.kernel[..., i], self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=1
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def emd2d(self, x, st_element, strides, padding, rates=(1, 1, 1, 1)):
        x1 = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", rates)
        x2 = tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", rates)
        return (x1 + x2) / 2

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class QuadraticAverage2D(Layer):
    """
    Quadratic Average between Dilation and Erosion 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        bias_initializer="Ones",
        bias_constraint=None,
        bias_regularization=None,
        scale=1,
        **kwargs,
    ):
        super(QuadraticAverage2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.scale = scale
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.bias_regularization = tf.keras.regularizers.get(bias_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]

        data = np.ones(self.kernel_size)
        data[int(data.shape[0] / 2), int(data.shape[1] / 2)] = 0
        # data=snm.distance_transform_edt(data)**2
        data = snm.distance_transform_edt(data)
        data = -((data / (2 * self.scale)) ** 2)
        data = np.repeat(data[:, :, np.newaxis], input_dim, axis=2)
        data = np.repeat(data[:, :, :, np.newaxis], self.num_filters, axis=3)
        self.data = tf.convert_to_tensor(data, np.float32)
        self.bias = self.add_weight(
            shape=(input_dim, self.num_filters),
            initializer=self.bias_initializer,
            name="bias",
            constraint=self.bias_constraint,
            regularizer=self.bias_regularization,
        )
        super(QuadraticAverage2D, self).build(input_shape)

    def emd2d(self, x, st_element, strides, padding, rates=(1, 1)):
        x1 = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        x2 = tf.nn.erosion2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        return (x1 + x2) / 2

    def call(self, x):
        kernel = tf.math.multiply(self.data, self.bias)
        for i in range(self.num_filters):
            out = self.emd2d(x, kernel[..., i], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
=======
Probing
=======
"""


class Probing2D(Layer):
    """
    Morphological Probing 2D Layer for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(Probing2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters) + (2,)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        super(Probing2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = self.probing2d(x, self.kernel[..., i, :], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def probing2d(self, x, st_element, strides, padding, rates=(1, 1)):
        x = tf.nn.dilation2d(
            x, st_element[..., 0], (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,)
        ) - tf.nn.erosion2d(x, st_element[..., 1], (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


"""
========
Gradient
========
"""


class Gradient2D(Layer):
    """
    Morphological Gradient 2D Layer for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(Gradient2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        super(Gradient2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = gradient2d(x, self.kernel[..., i], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class InternalGradient2D(Layer):
    """
    Internal Morphological Gradient 2D Layer for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(InternalGradient2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        super(InternalGradient2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = internalgradient2d(x, self.kernel[..., i], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class ExternalGradient2D(Layer):
    """
    External Morphological Gradient 2D Layer for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(ExternalGradient2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularization,
        )

        super(ExternalGradient2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = externalgradient2d(x, self.kernel[..., i], self.strides, self.padding, self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
            }
        )
        return config


class ToggleMapping2D(Layer):
    """
    ToggleMapping 2D Layer for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        steps=1,
        strides=(1, 1),
        padding="same",
        kernel_initializer="Zeros",
        kernel_constraint=None,
        kernel_regularization=None,
        **kwargs,
    ):
        super(ToggleMapping2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = (1, 1)
        self.steps = steps
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

        def build(self, input_shape):
            if input_shape[self.channel_axis] is None:
                raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")

            input_dim = input_shape[self.channel_axis]
            kernel_shape = self.kernel_size + (input_dim, self.num_filters)
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel",
                constraint=self.kernel_constraint,
                regularizer=self.kernel_regularization,
            )
            super(ToggleMapping2D, self).build(input_shape)

        def call(self, x):
            for i in range(self.num_filters):
                out = togglemapping2d(x, self.kernel[..., i], self.strides, self.padding, self.rates, self.steps)
                if i == 0:
                    outputs = out
                else:
                    outputs = K.concatenate([outputs, out])
                return outputs

        def compute_output_shape(self, input_shape):
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i], dilation=self.rates[i]
                )
            new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

        def get_config(self):
            config = super(ToggleMapping2D, self).get_config().copy()
            config.update(
                {
                    "num_filters": self.num_filters,
                    "kernel_size": self.kernel_size,
                    "strides": self.strides,
                    "padding": self.padding,
                    "dilation_rate": self.rates,
                }
            )
            return config


"""
==============
POOLING LAYERS
==============
"""


class MinPooling2D(Layer):
    """
    Min Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
    Arguments:
      pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """

    def __init__(self, pool_size, strides, padding="valid", data_format=None, name=None, **kwargs):
        super(MinPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        if self.data_format == "channels_last":
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = -nn.max_pool(
            -inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        return outputs

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(MinPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GradPooling2D(Layer):
    """
    Grad Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
    Arguments:
      pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """

    def __init__(self, pool_size, strides, padding="valid", data_format=None, name=None, **kwargs):
        super(GradPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        if self.data_format == "channels_last":
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = nn.max_pool(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        ) + nn.max_pool(
            -inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        return outputs

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(GradPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixedPooling2D(Layer):
    """
    Mixed Pooling.
    Combine max pooling and average pooling in fixed proportion specified by alpha a:
    f mixed (x) = a * f max(x) + (1-a) * f avg(x)
    Arguments:
      pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """

    def __init__(self, pool_size, strides, padding="valid", data_format=None, name=None, **kwargs):
        super(MixedPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.alpha = tf.Variable(initial_value=tf.ones(shape=(1,), dtype="float32") / 2, trainable=True)

    def call(self, inputs):
        if self.data_format == "channels_last":
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        output = (self.alpha) * nn.max_pool(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        ) + (1 - self.alpha) * nn.avg_pool2d(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        return output

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(MixedPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixedMaxMinPooling2D(Layer):
    """
    MixedMaxMinPooling2D.
    Combine max pooling and min pooling in fixed proportion specified by alpha a:
    f mixed (x) = a * f max(x) + (1-a) * f min(x)
    Arguments:
      pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """

    def __init__(self, pool_size, strides, padding="valid", data_format=None, name=None, **kwargs):
        super(MixedMaxMinPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.alpha = tf.Variable(initial_value=tf.ones(shape=(1,), dtype="float32") / 2, trainable=True)

    def call(self, inputs):
        if self.data_format == "channels_last":
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        output = (self.alpha) * nn.max_pool(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        ) + (1 - self.alpha) * -nn.max_pool2d(
            -inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        return output

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(MixedMaxMinPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LasryLionsDE(Layer):
    """
    Lasry Dilation 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        erosion_rate=(1, 1),
        bias_initializer=tf.keras.initializers.RandomUniform(minval=0.75, maxval=1.25),
        bias_initializer_bis=tf.keras.initializers.RandomUniform(minval=0.5, maxval=0.9),
        bias_constraint=None,
        bias_constraint_bis=ZeroToOne(),
        bias_regularization=None,
        bias_regularization_bis=None,
        scale=5,
        scale_bis=5,
        **kwargs,
    ):
        super(LasryLionsDE, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.rates_bis = erosion_rate
        self.scale = scale
        self.scale_bis = scale_bis
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.bias_regularization = tf.keras.regularizers.get(bias_regularization)
        self.bias_initializer_bis = tf.keras.initializers.get(bias_initializer_bis)
        self.bias_constraint_bis = tf.keras.constraints.get(bias_constraint_bis)
        self.bias_regularization_bis = tf.keras.regularizers.get(bias_regularization_bis)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")
        input_dim = input_shape[self.channel_axis]
        # pour dilation
        data = np.ones(self.kernel_size)
        data[int(data.shape[0] / 2), int(data.shape[1] / 2)] = 0
        # data=snm.distance_transform_edt(data)**2
        data = snm.distance_transform_edt(data)

        data1 = -((data / (4 * self.scale)) ** 2)
        data1 = np.repeat(data1[:, :, np.newaxis], input_dim, axis=2)
        data1 = np.repeat(data1[:, :, :, np.newaxis], self.num_filters, axis=3)
        self.data = tf.convert_to_tensor(data1, np.float32)

        # data1_bis=(-(data/(4*self.scale_bis))**2)
        # data1_bis = np.repeat(data1_bis[:, :, np.newaxis], input_dim, axis=2)
        # data1_bis = np.repeat(data1_bis[:, :, :,np.newaxis], self.num_filters, axis=3)
        # self.data_bis=tf.convert_to_tensor(data1_bis, np.float32)

        self.bias = self.add_weight(
            shape=(input_dim, self.num_filters),
            initializer=self.bias_initializer,
            constraint=self.bias_constraint,
            regularizer=self.bias_regularization,
            trainable=True,
        )
        self.bias_bis = self.add_weight(
            shape=(input_dim, self.num_filters),
            initializer=self.bias_initializer_bis,
            constraint=self.bias_constraint_bis,
            regularizer=self.bias_regularization_bis,
            trainable=True,
        )

        super(LasryLionsDE, self).build(input_shape)

    def call(self, x):
        kernel = tf.math.multiply(self.data, self.bias)
        kernel_c = tf.math.multiply(self.data, self.bias * self.bias_bis)
        for i in range(self.num_filters):
            y = self.dilation2d(x, kernel_c[..., i], self.strides, self.padding, self.rates)
            out = self.erosion2d(y, kernel[..., i], self.strides, self.padding, self.rates_bis)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i],
                erosion=self.rates_bis[i],
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
                "erosion_rate": self.rates_bis,
            }
        )
        return config


class LasryLionsED(Layer):
    """
    Lasry Erosion-Dilation 2D Layer
    for now assuming channel last
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        erosion_rate=(1, 1),
        bias_initializer=tf.keras.initializers.RandomUniform(minval=0.75, maxval=1.25),
        bias_initializer_bis=tf.keras.initializers.RandomUniform(minval=0.5, maxval=0.9),
        bias_constraint=None,
        bias_constraint_bis=ZeroToOne(),
        bias_regularization=None,
        bias_regularization_bis=None,
        scale=5,
        scale_bis=5,
        **kwargs,
    ):
        super(LasryLionsED, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate
        self.rates_bis = erosion_rate
        self.scale = scale
        self.scale_bis = scale_bis
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.bias_regularization = tf.keras.regularizers.get(bias_regularization)
        self.bias_initializer_bis = tf.keras.initializers.get(bias_initializer_bis)
        self.bias_constraint_bis = tf.keras.constraints.get(bias_constraint_bis)
        self.bias_regularization_bis = tf.keras.regularizers.get(bias_regularization_bis)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")
        input_dim = input_shape[self.channel_axis]
        # pour dilation
        data = np.ones(self.kernel_size)
        data[int(data.shape[0] / 2), int(data.shape[1] / 2)] = 0
        # data=snm.distance_transform_edt(data)**2
        data = snm.distance_transform_edt(data)

        data1 = -((data / (4 * self.scale)) ** 2)
        data1 = np.repeat(data1[:, :, np.newaxis], input_dim, axis=2)
        data1 = np.repeat(data1[:, :, :, np.newaxis], self.num_filters, axis=3)
        self.data = tf.convert_to_tensor(data1, np.float32)

        # data1_bis=(-(data/(4*self.scale_bis))**2)
        # data1_bis = np.repeat(data1_bis[:, :, np.newaxis], input_dim, axis=2)
        # data1_bis = np.repeat(data1_bis[:, :, :,np.newaxis], self.num_filters, axis=3)
        # self.data_bis=tf.convert_to_tensor(data1_bis, np.float32)

        self.bias = self.add_weight(
            shape=(input_dim, self.num_filters),
            initializer=self.bias_initializer,
            constraint=self.bias_constraint,
            regularizer=self.bias_regularization,
            trainable=True,
        )
        self.bias_bis = self.add_weight(
            shape=(input_dim, self.num_filters),
            initializer=self.bias_initializer_bis,
            constraint=self.bias_constraint_bis,
            regularizer=self.bias_regularization_bis,
            trainable=True,
        )

        super(LasryLionsED, self).build(input_shape)

    def call(self, x):
        kernel = tf.math.multiply(self.data, self.bias)
        kernel_c = tf.math.multiply(self.data, self.bias * self.bias_bis)
        for i in range(self.num_filters):
            y = self.dilation2d(x, kernel_c[..., i], self.strides, self.padding, self.rates)
            out = self.erosion2d(y, kernel[..., i], self.strides, self.padding, self.rates_bis)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i],
                erosion=self.rates_bis[i],
            )
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters * input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.rates,
                "erosion_rate": self.rates_bis,
            }
        )
        return config


"""
==========================================
Morphological Empirical Mode Decomposition
==========================================
"""


def MorphoEMP2D(input_layer, num_filters, kernel_size, strides):
    xP = Dilation2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=MinusOnesZeroCenter(),
        trainable=False,
        padding="valid",
    )(input_layer)
    xd = Dilation2D(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer="Zeros", padding="valid")(
        input_layer
    )
    xe = Erosion2D(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer="Zeros", padding="valid")(
        input_layer
    )
    xc = (xd + xe) / 2
    return xP - xc


def MorphoEMP2DShare(input_layer, num_filters, kernel_size, strides):
    xP = Dilation2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=MinusOnesZeroCenter(),
        trainable=False,
        padding="valid",
    )(input_layer)
    xc = MorphoAverage2D(num_filters, kernel_size=kernel_size, strides=strides, padding="valid")(input_layer)
    return xP - xc


def MorphoEMD2DQuadratic(input_layer, num_filters, kernel_size, strides):
    xP = Dilation2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=MinusOnesZeroCenter(),
        trainable=False,
        padding="valid",
    )(input_layer)
    xd = QuadraticDilation2D(num_filters, kernel_size=kernel_size, strides=strides, padding="valid")(input_layer)
    xe = QuadraticDilation2D(num_filters, kernel_size=kernel_size, strides=strides, padding="valid")(-input_layer)
    xc = (xd - xe) / 2
    return xP - xc


def MorphoEMP2DQuadraticShare(input_layer, num_filters, kernel_size, strides):
    xP = Dilation2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=MinusOnesZeroCenter(),
        trainable=False,
        padding="valid",
    )(input_layer)
    xc = QuadraticAverage2D(num_filters, kernel_size=kernel_size, strides=strides, padding="valid")(input_layer)
    return xP - xc


def SeparableOperator2D(
    x,
    num_filters,
    kernel_size,
    operator=dilation2d,
    integrator=tf.reduce_sum,
    strides=(1, 1),
    padding="same",
    dilation_rate=(1, 1),
    kernel_initializer="Zeros",
    kernel_constraint=None,
    kernel_regularization=None,
    bias_initializer="zeros",
    bias_regularizer=None,
    bias_constraint=None,
    shared=False,
    trainable=True,
):
    x = DepthwiseOperator2D(
        kernel_size=kernel_size,
        operator=operator,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_constraint=kernel_constraint,
        kernel_regularization=kernel_regularization,
        shared=shared,
        trainable=trainable,
    )(x)
    x = IntegratorofOperator2D(
        num_filters=num_filters,
        kernel_size=(1, 1),
        operator=operator,
        integrator=integrator,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        bias_constraint=bias_constraint,
    )(x)
    return x


def BSOperator2D(
    x,
    num_filters,
    kernel_size,
    operator=dilation2d,
    integrator=tf.reduce_sum,
    strides=(1, 1),
    padding="same",
    dilation_rate=(1, 1),
    kernel_initializer="Zeros",
    kernel_constraint=None,
    kernel_regularization=None,
    bias_initializer="zeros",
    bias_regularizer=None,
    bias_constraint=None,
    shared=False,
    trainable=True,
):
    x = IntegratorofOperator2D(
        num_filters=num_filters, kernel_size=(1, 1), operator=operator, integrator=integrator, use_bias=False
    )(x)
    x = DepthwiseOperator2D(
        kernel_size=kernel_size,
        operator=operator,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_constraint=kernel_constraint,
        kernel_regularization=kernel_regularization,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        bias_constraint=bias_constraint,
        shared=shared,
        trainable=trainable,
    )(x)
    return x
