"""Defines the network architectures."""
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
import numpy as np
from typing import List
import logging
try:
    import tcn as tcn_layer
except ImportError:
    logging.warning('Could not import TCN layer.')


model_dict = dict()

def register_as_model(func):
    """Adds func to model_dict Dict[modelname: modelfunc]. For selecting models by string."""
    model_dict[func.__name__] = func
    return func


class ModelMGPU(keras.models.Model):
    """Convert keras model to run on multiple GPUs.

    from here: https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
    could also try: https://github.com/keras-team/keras/issues/11253#issuecomment-458788861
    """

    def __init__(self, ser_model: keras.models.Model, gpus: int):
        """Make multi-GPU model from simple keras Model.
        
        Args:
            ser_model (keras.models.Model): The single-cpu/gpu keras model.
            gpus (int): Number of GPUs to init the multi-GPU model for.
        """
        try:
            pmodel = keras.utils.multi_gpu_model(ser_model, gpus)
            pmodel.compile(ser_model.optimizer, loss=ser_model.loss)
        except ValueError:
            logging.info('Could not create multi GPU model. Will train on single GPU.', exc_info=False)
            pmodel = ser_model
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        """Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        """
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


@register_as_model
def cnn(nb_freq, nb_classes, nb_channels=1, nb_hist=1, nb_filters=16,
        nb_stacks=2, kernel_size=3, nb_conv=3, loss="categorical_crossentropy",
        batch_norm=False, return_sequences=False, sample_weight_mode: str = None,
        **kwignored):
    """CNN for single-frequency and multi-channel data - uses 1D convolutions.
    
    Args:
        nb_freq ([type]): [description]
        nb_classes ([type]): [description]
        nb_channels (int, optional): [description]. Defaults to 1.
        nb_hist (int, optional): [description]. Defaults to 1.
        nb_filters (int, optional): [description]. Defaults to 16.
        nb_stacks (int, optional): [description]. Defaults to 2.
        kernel_size (int, optional): [description]. Defaults to 3.
        nb_conv (int, optional): [description]. Defaults to 3.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        batch_norm (bool, optional): [description]. Defaults to False.
        return_sequences (bool, optional): [description]. Defaults to False.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred
    
    Raises:
        ValueError: When trying to init model with return_sequences.
                    The cnn only does instance classificaton, not regression (sequence prediction).
    
    Returns:
        [type]: [description]
    """
    if return_sequences:
        raise ValueError('Cannot perform regression with CNN.')
    inp = kl.Input(shape=(nb_hist, nb_channels))
    out = inp
    for conv in range(nb_conv):
        for _ in range(nb_stacks):
            out = kl.Conv1D(nb_filters * (2 ** conv), kernel_size, padding='same', activation='relu')(out)
            out = kl.BatchNormalization()(out) if batch_norm else out
        out = kl.MaxPooling1D(min(int(out.shape[1]), 2))(out)

    out = kl.Flatten()(out)
    out = kl.Dense(nb_classes * 4, activation='relu')(out)
    out = kl.Dense(nb_classes * 2, activation='relu')(out)
    out = kl.Dense(nb_classes, activation='relu')(out)
    out = kl.Activation("softmax")(out)

    model = keras.models.Model(inp, out, name='FCN')
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, amsgrad=True),
                  loss=loss, sample_weight_mode=sample_weight_mode)
    return model


@register_as_model
def cnn2D(nb_freq, nb_classes, nb_channels=1, nb_hist=1, nb_filters=16,
          nb_stacks=2, kernel_size=3, nb_conv=3, loss="categorical_crossentropy",
          batch_norm=False, return_sequences=False, sample_weight_mode: str = None,
          **kwignored):
    """CNN for multi-frequency and multi-channel data - uses 2D convolutions.
    
    Args:
        nb_freq ([type]): [description]
        nb_classes ([type]): [description]
        nb_channels (int, optional): [description]. Defaults to 1.
        nb_hist (int, optional): [description]. Defaults to 1.
        nb_filters (int, optional): [description]. Defaults to 16.
        nb_stacks (int, optional): [description]. Defaults to 2.
        kernel_size (int, optional): [description]. Defaults to 3.
        nb_conv (int, optional): [description]. Defaults to 3.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        batch_norm (bool, optional): [description]. Defaults to False.
        return_sequences (bool, optional): [description]. Defaults to False.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred
    
    Raises:
        ValueError: When trying to init model with return_sequences.
                    The cnn only does instance classificaton, not regression (sequence prediction).
    
    Returns:
        [type]: [description]
    """
    inp = kl.Input(shape=(nb_hist, nb_freq, nb_channels))
    out = inp
    for conv in range(nb_conv):
        for stack in range(nb_stacks):
            out = kl.Conv2D(nb_filters * (2 ** conv), (max(1, int(out.shape[1])), kernel_size), padding='same', activation='relu')(out)
            out = kl.BatchNormalization()(out) if batch_norm else out
        # out = Conv2D(nb_filters * (2 ** conv), (max(1, int(out.shape[1])), kernel_size), padding='same', activation='relu')(out)
        # out = BatchNormalization()(out) if batch_norm else out
        out = kl.MaxPooling2D((min(int(out.shape[1]), 2), 2))(out)
    out = kl.Flatten()(out)
    out = kl.Dense(int(nb_classes * 8 * nb_filters), activation='relu')(out)
    out = kl.Dropout(0.1)(out)
    out = kl.Dense(int(nb_classes * 4 * nb_filters), activation='relu')(out)
    out = kl.Dropout(0.1)(out)
    out = kl.Dense(int(nb_classes * 2 * nb_filters), activation='relu')(out)
    out = kl.Dropout(0.1)(out)
    out = kl.Dense(nb_classes, activation='relu')(out)
    out = kl.Activation("softmax")(out)

    model = keras.models.Model(inp, out, name='CNN2D')
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, amsgrad=True),
                  loss=loss, sample_weight_mode=sample_weight_mode)
    return model


@register_as_model
def fcn(nb_freq, nb_classes, nb_channels=1, nb_hist=1, nb_filters=16,
        nb_stacks=2, kernel_size=3, nb_conv=3, loss="categorical_crossentropy",
        batch_norm=False, return_sequences=True, sample_weight_mode: str = None,
        **kwignored):
    """[summary]
    
    Args:
        nb_freq ([type]): [description]
        nb_classes ([type]): [description]
        nb_channels (int, optional): [description]. Defaults to 1.
        nb_hist (int, optional): [description]. Defaults to 1.
        nb_filters (int, optional): [description]. Defaults to 16.
        nb_stacks (int, optional): [description]. Defaults to 2.
        kernel_size (int, optional): [description]. Defaults to 3.
        nb_conv (int, optional): [description]. Defaults to 3.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        batch_norm (bool, optional): [description]. Defaults to False.
        return_sequences (bool, optional): [description]. Defaults to True.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred
    
    Returns:
        [type]: [description]
    """
    inp = kl.Input(shape=(nb_hist, nb_channels))
    out = inp
    for conv in range(nb_conv):
        for _ in range(nb_stacks):
            out = kl.Conv1D(nb_filters * (2 ** conv), kernel_size,
                            padding='same', activation='relu')(out)
            out = kl.BatchNormalization()(out) if batch_norm else out
        out = kl.MaxPooling1D(min(int(out.shape[1]), 2))(out)

    for conv in range(nb_conv, 0, -1):
        out = kl.UpSampling1D(size=2)(out)
        out = kl.Conv1D(nb_filters * (2 ** conv), kernel_size,
                        padding='same', activation='relu')(out)

    if not return_sequences:
        out = kl.Flatten()(out)
    out = kl.Dense(nb_classes)(out)
    out = kl.Activation("softmax")(out)

    model = keras.models.Model(inp, out, name='FCN')
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, amsgrad=True),
                  loss=loss, sample_weight_mode=sample_weight_mode)
    return model


@register_as_model
def fcn2D(nb_freq, nb_classes, nb_channels=1, nb_hist=1, nb_filters=16,
          nb_stacks=2, kernel_size=3, nb_conv=3, loss="categorical_crossentropy",
          batch_norm=False, return_sequences=True, sample_weight_mode: str = None,
          **kwignored):
    """[summary]
    
    Args:
        nb_freq ([type]): [description]
        nb_classes ([type]): [description]
        nb_channels (int, optional): [description]. Defaults to 1.
        nb_hist (int, optional): [description]. Defaults to 1.
        nb_filters (int, optional): [description]. Defaults to 16.
        nb_stacks (int, optional): [description]. Defaults to 2.
        kernel_size (int, optional): [description]. Defaults to 3.
        nb_conv (int, optional): [description]. Defaults to 3.
        loss (str, optional): [description]. Defaults to "categorical_crossentropy".
        batch_norm (bool, optional): [description]. Defaults to False.
        return_sequences (bool, optional): [description]. Defaults to True.
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred
    
    Returns:
        [type]: [description]
    """
    inp = kl.Input(shape=(nb_hist, nb_freq, nb_channels))
    out = inp
    for conv in range(nb_conv):
        for stack in range(nb_stacks):
            out = kl.Conv2D(nb_filters * (2 ** conv), (max(1, int(out.shape[1])), kernel_size), padding='same', activation='relu')(out)
            out = kl.BatchNormalization()(out) if batch_norm else out
        out = kl.MaxPooling2D((min(int(out.shape[1]), 2), 2))(out)
    for conv in range(nb_conv, 0, -1):
            out = kl.Conv2D(int(nb_filters * (conv / 2)), (max(1, int(out.shape[1])), kernel_size), padding='same', activation='relu')(out)
            out = kl.BatchNormalization()(out) if batch_norm else out

    out = kl.Flatten()(out)
    out = kl.Dense(nb_classes, activation='relu')(out)
    out = kl.Activation("softmax")(out)

    model = keras.models.Model(inp, out, name='CNN')
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0005, amsgrad=True), loss=loss)
    return model


@register_as_model
def tcn_seq(nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
            nb_conv: int = 1, loss: str = "categorical_crossentropy",
            dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
            use_skip_connections: bool = True, return_sequences: bool = True,
            dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
            **kwignored):
    """Create TCN network.

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
        kwignored (Dict, optional): additional kw args in the param dict used for calling m(**params) to be ingonred

    Returns:
        [keras.models.Model]: Compiled TCN network model.
    """
    # TODO: rename to tcn
    dilations = tcn_layer.tcn.process_dilations(dilations)
    input_layer = kl.Input(shape=(nb_hist, nb_freq))

    # # TCN(nb_filters=64, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8, 16, 32], padding='causal',
    # # use_skip_connections=True, dropout_rate=0.0, return_sequences=True, activation='linear', name='tcn', kernel_initializer='he_normal', use_batch_norm=False)
    # x = tcn.TCN(nb_filters, kernel_size, nb_conv, dilations, padding, use_skip_connections, dropout_rate, return_sequences, activation, )(input_layer)
    x = tcn_layer.TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_conv, dilations=dilations, activation=activation,
                      use_skip_connections=use_skip_connections, padding=padding, dropout_rate=dropout_rate, return_sequences=return_sequences)(input_layer)
    x = kl.Dense(nb_classes)(x)
    x = kl.Activation('softmax')(x)
    output_layer = x
    model = keras.models.Model(input_layer, output_layer, name='TCN')
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, amsgrad=True, clipnorm=1.),
                  loss=loss, sample_weight_mode=sample_weight_mode)
    return model


@register_as_model
def tcn(*args, **kwargs):
    """Synonym for tcn_seq."""
    return tcn_seq(*args, **kwargs)
