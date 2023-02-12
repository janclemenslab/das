import tensorflow.keras.backend as K
import tensorflow.keras.layers
import tensorflow.keras as keras
from tensorflow.keras import optimizers

# from tensorflow.keras.engine.topology import Layer  #only used for type annotations
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, SeparableConv1D
from tensorflow.keras.layers import Conv1D, Dense, Layer
from tensorflow.keras import Input, Model

from typing import List, Tuple


def channel_normalization(x):
    # type: (Layer) -> Layer
    """Normalize a layer to the maximum activation

    This keeps a layers values between zero and one.
    It helps with relu's unbounded activation

    Args:
        x: The layer to normalize

    Returns:
        A maximal normalized layer
    """
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x: Layer) -> Layer:
    """This method defines the activation used for WaveNet

    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/

    Args:
        x: The layer we want to apply the activation to

    Returns:
        A new layer with the wavenet activation applied
    """
    tanh_out = Activation("tanh")(x)
    sigm_out = Activation("sigmoid")(x)
    return keras.layers.multiply([tanh_out, sigm_out])


def residual_block(
    x: Layer,
    s: int,
    i: int,
    activation: str,
    nb_filters: int,
    kernel_size: int,
    padding: str = "causal",
    use_separable: bool = False,
    dropout_rate: float = 0,
    name: str = "",
) -> Tuple[Layer, Layer]:
    """Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        s: The stack index i.e. which stack in the overall TCN
        i: The dilation power of 2 we are using for this residual block
        activation: The name of the type of activation to use
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        use_separable: Use separable convolution
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.

    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """

    original_x = x

    if use_separable:
        conv = SeparableConv1D(
            filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, depth_multiplier=4, padding=padding
        )(x)
    else:
        conv = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding=padding)(x)

    if activation == "norm_relu":
        x = Activation("relu")(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == "wavenet":
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(dropout_rate)(x)

    # 1x1 conv.
    x = Conv1D(nb_filters, 1, padding="same")(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2**i for i in dilations]
        return new_dilations


class TCN:
    """Creates a TCN layer.

    Args:
        input_layer: A tensor of shape (batch_size, timesteps, input_dim).
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        activation: The activations to use (norm_relu, wavenet, relu...).
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
        use_separable: Boolean. Use separable convolutions in each residual block.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        padding: The padding to use in the convolutional layers, 'causal' or 'same'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.

    Returns:
        A TCN layer.
    """

    def __init__(
        self,
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=None,
        activation="norm_relu",
        use_skip_connections=True,
        use_separable=False,
        padding="causal",
        dropout_rate=0.0,
        return_sequences=True,
        name="tcn",
    ):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections

        try:
            len(use_separable)
        except TypeError:
            use_separable = [use_separable] * nb_stacks

        while len(use_separable) < nb_stacks:
            use_separable.append(use_separable[-1])
        self.use_separable = use_separable

        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        x = inputs
        x = Conv1D(self.nb_filters, 1, padding=self.padding)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(
                    x,
                    s,
                    i,
                    self.activation,
                    self.nb_filters,
                    self.kernel_size,
                    self.padding,
                    self.use_separable[s],
                    self.dropout_rate,
                )
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        x = Activation("relu")(x)

        if not self.return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x
