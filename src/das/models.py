"""Defines the network architectures."""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from typing import List, Optional, Tuple
from . import tcn as tcn_layer
from .kapre.time_frequency import Spectrogram
from . import spec_utils
from .loss import TMSE, WeightedLoss
from das.morpholayers.layers import Closing2D, Opening2D
from das.morpholayers.regularizers import l1lattice

model_dict = dict()


def _register_as_model(func):
    """Adds func to model_dict Dict[modelname: modelfunc]. For selecting models by string."""
    model_dict[func.__name__] = func
    return func


@_register_as_model
def tcn(*args, **kwargs):
    """Synonym for tcn_stft."""
    return tcn_stft(*args, **kwargs)


@_register_as_model
def tcn_stft(
    nb_freq: int,
    nb_classes: int,
    nb_hist: int = 1,
    nb_filters: int = 16,
    kernel_size: int = 3,
    nb_conv: int = 1,
    loss: str = "categorical_crossentropy",
    dilations: Optional[List[int]] = None,
    activation: str = "norm_relu",
    use_skip_connections: bool = True,
    return_sequences: bool = True,
    dropout_rate: float = 0.00,
    padding: str = "same",
    sample_weight_mode: str = None,
    nb_pre_conv: int = 0,
    pre_nb_dft: int = 64,
    nb_lstm_units: int = 0,
    morph_nb_kernels: int = 1,
    morph_kernel_duration: int = 32,
    learning_rate: float = 0.0005,
    upsample: bool = True,
    use_separable: bool = False,
    use_resnet: bool = False,
    tmse_weight: float = 0.0,
    compile: bool = True,
    **kwignored,
):
    """Create TCN network with optional trainable STFT layer as pre-processing and downsampling frontend.

    Args:
        nb_freq (int): [description]
        nb_classes (int): [description]
        nb_hist (int, optional): [description]. Defaults to 1.
        nb_filters (int, optional): [description]. Defaults to 16.
        kernel_size (int, optional): [description]. Defaults to 3.
        nb_conv (int, optional): [description]. Defaults to 1.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        dilations (List[int], optional): [description]. Defaults to [1, 2, 4, 8, 16].
        activation (str, optional): [description]. Defaults to 'norm_relu'.
        use_skip_connections (bool, optional): [description]. Defaults to True.
        return_sequences (bool, optional): [description]. Defaults to True.
        dropout_rate (float, optional): [description]. Defaults to 0.00.
        padding (str, optional): [description]. Defaults to 'same'.
        nb_pre_conv (int, optional): If >0 adds a single STFT layer with a hop size of 2**nb_pre_conv before the TCN.
                                     Useful for speeding up training by reducing the sample rate early in the network.
                                     Defaults to 0 (no downsampling)
        pre_nb_dft (int, optional): Duration of filters (in samples) for the STFT frontend.
                                    Number of filters is pre_nb_dft // 2 + 1.
                                    Defaults to 64.
        learning_rate (float, optional) Defaults to 0.0005
        nb_lstm_units (int, optional): Defaults to 0.
        morph_nb_kernels (int): Defaults to 0 (no morphological kernels).
        morph_kernel_duration (int): Defaults to 32.
        upsample (bool, optional): whether or not to restore the model output to the input samplerate.
                                   Should generally be True during training and evaluation but may speed up inference.
                                   Defaults to True.
        use_separable (bool, optional): use separable convs in residual block. Defaults to False.
        use_resnet (bool, optional): Defaults to False.
        tmse_weight (float): Defaults to 0.0. No tmse.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred

    Returns:
        [keras.models.Model]: Compiled TCN network model.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    input_layer = kl.Input(shape=(nb_hist, nb_freq))
    out = input_layer

    if nb_pre_conv > 0:
        out = Spectrogram(
            n_dft=pre_nb_dft,
            n_hop=2**nb_pre_conv,
            return_decibel_spectrogram=True,
            power_spectrogram=1.0,
            trainable_kernel=True,
            name="trainable_stft",
            image_data_format="channels_last",
        )(out)
        if not use_resnet:
            out = kl.Reshape((out.shape[1], out.shape[2] * out.shape[3]))(out)

    if use_resnet:
        out = kl.Activation("relu")(out)
        out = kl.Concatenate(axis=-1)([out, out, out])
        out = ResNet50V2(input_shape=out.shape[1:], weights="imagenet", include_top=False)(out)
        out = kl.Reshape((out.shape[1], out.shape[2] * out.shape[3]))(out)
        out = kl.BatchNormalization()(out)

    x = tcn_layer.TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_conv,
        dilations=dilations,
        activation=activation,
        use_skip_connections=use_skip_connections,
        padding=padding,
        dropout_rate=dropout_rate,
        return_sequences=return_sequences,
        use_separable=use_separable,
    )(out)

    if nb_lstm_units > 0:
        x = kl.Bidirectional(kl.LSTM(units=nb_lstm_units, return_sequences=True))(x)

    x = kl.Dense(nb_classes, activation="softmax")(x)

    if morph_nb_kernels > 0:
        x = x[:, :, tf.newaxis, :]
        x = Closing2D(
            num_filters=morph_nb_kernels,
            padding="same",
            kernel_size=(morph_kernel_duration, 1),
            kernel_regularization=l1lattice(0.002),
            # kernel_constraint=ser,
        )(x)
        x = Opening2D(
            num_filters=morph_nb_kernels,
            padding="same",
            kernel_size=(morph_kernel_duration, 1),
            kernel_regularization=l1lattice(0.002),
            # kernel_constraint=ser,
        )(x)
        x = x[..., 0, :]

    if nb_pre_conv > 0 and upsample:
        x = kl.UpSampling1D(size=2**nb_pre_conv)(x)

    output_layer = x

    model = keras.models.Model(input_layer, output_layer, name="TCN")
    if use_resnet:
        model.get_layer(name="trainable_stft").trainable = False
        model.get_layer(name="resnet50v2").trainable = False

    if tmse_weight > 0:
        batch_size = 32
        loss = WeightedLoss([tf.keras.metrics.categorical_crossentropy, TMSE(batch_size)], [1.0, tmse_weight])

    if compile:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode=sample_weight_mode)
    return model


@_register_as_model
def stft_res_dense(
    nb_freq: int,
    nb_classes: int,
    nb_hist: int = 1,
    sample_weight_mode: str = None,
    learning_rate: float = 0.0005,
    compile: bool = True,
    stft_compute: bool = False,
    resnet_compute: bool = False,
    resnet_train: bool = False,
    label_smoothing: float = 0,
    **kwignored,
):
    """Create TCN network with optional trainable STFT layer as pre-processing and downsampling frontend.

    Args:
        nb_freq (int): [description]
        nb_classes (int): [description]
        nb_hist (int, optional): [description]. Defaults to 1.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        nb_pre_conv (int, optional): If >0 adds a single STFT layer with a hop size of 2**nb_pre_conv before the TCN.
                                     Useful for speeding up training by reducing the sample rate early in the network.
                                     Defaults to 0 (no downsampling)
        pre_nb_dft (int, optional): Duration of filters (in samples) for the STFT frontend.
                                    Number of filters is pre_nb_dft // 2 + 1.
                                    Defaults to 64.
        learning_rate (float, optional) Defaults to 0.0005
        stft_compute (bool, optional): Defaults to False.
        resnet_compute (bool, optional): Defaults to False.
        resnet_train (bool, optional): Fine tune resnet weights. Defaults to False.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred

    Returns:
        [keras.models.Model]: Compiled network model.
    """

    input_layer = kl.Input(shape=(nb_hist, nb_freq))
    out = input_layer

    # TODO compute STFT

    if resnet_compute:
        out = tf.stack((out, out, out), axis=-1)
        vision_model = ResNet50V2(input_shape=out.shape[1:], weights="imagenet", include_top=False)
        out = vision_model(out, training=False)
        out = kl.BatchNormalization()(out)
        out = kl.TimeDistributed(
            kl.Dense(min(32, 4 * nb_classes), activation="tanh", kernel_regularizer=regularizers.L1(1e-4))
        )(out)

    if len(out.shape) > 1:
        out = kl.Flatten()(out)

    # out = kl.BatchNormalization()(out)
    # out = kl.Dropout(0.1)(out)
    out = kl.Dense(min(32, 4 * nb_classes), activation="tanh", kernel_regularizer=regularizers.L1(1e-4))(out)
    # out = kl.Dropout(0.1)(out)
    out = kl.Dense(2 * nb_classes, activation="tanh", kernel_regularizer=regularizers.L1(1e-4))(out)
    # out = kl.Dropout(0.1)(out)
    out = kl.Dense(nb_classes, activation="softmax")(out)

    output_layer = out

    model = keras.models.Model(input_layer, output_layer, name="RES")

    if resnet_compute and not resnet_train:
        model.get_layer("resnet50v2").trainable = False

    if compile:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            sample_weight_mode=sample_weight_mode,
        )
    return model


@_register_as_model
def tcn_stft_morph(
    nb_freq: int,
    nb_classes: int,
    nb_hist: int = 1,
    nb_filters: int = 16,
    kernel_size: int = 3,
    nb_conv: int = 1,
    loss: str = "categorical_crossentropy",
    dilations: Optional[List[int]] = None,
    activation: str = "norm_relu",
    use_skip_connections: bool = True,
    return_sequences: bool = True,
    dropout_rate: float = 0.00,
    padding: str = "same",
    sample_weight_mode: str = None,
    nb_pre_conv: int = 0,
    pre_nb_dft: int = 64,
    nb_lstm_units: int = 0,
    morph_nb_kernels: int = 1,
    morph_kernel_duration: int = 33,
    learning_rate: float = 0.0005,
    upsample: bool = True,
    use_separable: bool = False,
    use_resnet: bool = False,
    compile: bool = True,
    **kwignored,
):
    """Create TCN network with optional trainable STFT layer as pre-processing and downsampling frontend.

    Args:
        nb_freq (int): [description]
        nb_classes (int): [description]
        nb_hist (int, optional): [description]. Defaults to 1.
        nb_filters (int, optional): [description]. Defaults to 16.
        kernel_size (int, optional): [description]. Defaults to 3.
        nb_conv (int, optional): [description]. Defaults to 1.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        dilations (List[int], optional): [description]. Defaults to [1, 2, 4, 8, 16].
        activation (str, optional): [description]. Defaults to 'norm_relu'.
        use_skip_connections (bool, optional): [description]. Defaults to True.
        return_sequences (bool, optional): [description]. Defaults to True.
        dropout_rate (float, optional): [description]. Defaults to 0.00.
        padding (str, optional): [description]. Defaults to 'same'.
        nb_pre_conv (int, optional): If >0 adds a single STFT layer with a hop size of 2**nb_pre_conv before the TCN.
                                     Useful for speeding up training by reducing the sample rate early in the network.
                                     Defaults to 0 (no downsampling)
        pre_nb_dft (int, optional): Duration of filters (in samples) for the STFT frontend.
                                    Number of filters is pre_nb_dft // 2 + 1.
                                    Defaults to 64.
        learning_rate (float, optional) Defaults to 0.0005
        nb_lstm_units (int, optional): Defaults to 0 (no lstm units).
        morph_nb_kernels (int): Defaults to 1 (no morphological kernels).
        morph_kernel_duration (int): Defaults to 33.
        upsample (bool, optional): whether or not to restore the model output to the input samplerate.
                                   Should generally be True during training and evaluation but may speed up inference.
                                   Defaults to True.
        use_separable (bool, optional): use separable convs in residual block. Defaults to False.
        use_resnet (bool, optional): Defaults to False.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred

    Returns:
        [keras.models.Model]: Compiled TCN network model.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    input_layer = kl.Input(shape=(nb_hist, nb_freq))
    out = input_layer

    if nb_pre_conv > 0:
        out = Spectrogram(
            n_dft=pre_nb_dft,
            n_hop=2**nb_pre_conv,
            return_decibel_spectrogram=True,
            power_spectrogram=1.0,
            trainable_kernel=True,
            name="trainable_stft",
            image_data_format="channels_last",
        )(out)
        if not use_resnet:
            out = kl.Reshape((out.shape[1], out.shape[2] * out.shape[3]))(out)

    if use_resnet:
        out = kl.Activation("relu")(out)
        out = kl.Concatenate(axis=-1)([out, out, out])
        out = ResNet50V2(input_shape=out.shape[1:], weights="imagenet", include_top=False)(out)
        out = kl.Reshape((out.shape[1], out.shape[2] * out.shape[3]))(out)
        out = kl.BatchNormalization()(out)

    x = tcn_layer.TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_conv,
        dilations=dilations,
        activation=activation,
        use_skip_connections=use_skip_connections,
        padding=padding,
        dropout_rate=dropout_rate,
        return_sequences=return_sequences,
        use_separable=use_separable,
    )(out)

    if nb_lstm_units > 0:
        x = kl.Bidirectional(kl.LSTM(units=nb_lstm_units, return_sequences=True))(x)

    x = kl.Dense(nb_classes, activation="softmax")(x)

    if morph_nb_kernels > 0:
        x = x[:, :, tf.newaxis, :]
        x = Closing2D(
            num_filters=morph_nb_kernels,
            padding="same",
            kernel_size=(morph_kernel_duration, 1),
            kernel_regularization=l1lattice(0.002),
            # kernel_constraint=ser,
        )(x)
        x = Opening2D(
            num_filters=morph_nb_kernels,
            padding="same",
            kernel_size=(morph_kernel_duration, 1),
            kernel_regularization=l1lattice(0.002),
            # kernel_constraint=ser,
        )(x)
        x = x[..., 0, :]

    if nb_pre_conv > 0 and upsample:
        x = kl.UpSampling1D(size=2**nb_pre_conv)(x)

    output_layer = x

    model = keras.models.Model(input_layer, output_layer, name="TCN")

    if use_resnet:
        model.get_layer(name="trainable_stft").trainable = False
        model.get_layer(name="resnet50v2").trainable = False

    if compile:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode=sample_weight_mode)
    return model
