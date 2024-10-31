from tensorflow.math import square
from tensorflow.nn import log_softmax
from tensorflow.raw_ops import Mean
from tensorflow.experimental.numpy import clip, append
from tensorflow.keras.losses import Loss
import tensorflow as tf


class TMSE(Loss):
    def __init__(self, batch_size, trunc=None):
        super().__init__()
        self.batch_size = batch_size
        self.trunc = trunc

    def call(self, y_true, y_pred):
        """Temporal MSE Loss

        Temporal MSE Loss Function
        Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
        arXiv: https://arxiv.org/pdf/1903.01945.pdf

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_
            trunc (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        squared_difference = square(log_softmax(y_pred[:, 1:], axis=1) - log_softmax(y_pred[:, :-1], axis=1))
        tmse = Mean(input=squared_difference, axis=-1)
        if self.trunc is not None:
            tmse = clip(tmse, a_min=0, a_max=self.trunc)
        tmse = append(tmse, tf.zeros((self.batch_size, 1)), axis=-1)

        return tmse


class WeightedLoss(Loss):
    def __init__(self, losses, loss_weights):
        super().__init__()
        self.losses = losses
        self.loss_weights = loss_weights

    def call(self, y_true, y_pred):
        total_loss = 0
        for loss, weight in zip(self.losses, self.loss_weights):
            total_loss += weight * loss(y_true, y_pred)
        return total_loss
