"""Backward-compatible custom losses."""

import keras


class TMSE(keras.losses.Loss):
    """Temporal MSE loss used by TensorFlow-backed DAS releases."""

    def __init__(self, batch_size, trunc=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.trunc = trunc

    def call(self, y_true, y_pred):
        del y_true
        difference = keras.ops.log_softmax(y_pred[:, 1:], axis=1) - keras.ops.log_softmax(y_pred[:, :-1], axis=1)
        tmse = keras.ops.mean(keras.ops.square(difference), axis=-1)
        if self.trunc is not None:
            tmse = keras.ops.clip(tmse, 0, self.trunc)
        return keras.ops.concatenate([tmse, keras.ops.zeros((self.batch_size, 1))], axis=-1)


class WeightedLoss(keras.losses.Loss):
    """Weighted sum of loss callables."""

    def __init__(self, losses, loss_weights, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        self.loss_weights = loss_weights

    def call(self, y_true, y_pred):
        return sum(weight * loss(y_true, y_pred) for loss, weight in zip(self.losses, self.loss_weights))
