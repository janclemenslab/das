from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import Regularizer

MIN_LATT = -1


class L1L2Lattice(Regularizer):
    """Regularizer for L1 and L2 regularization in a lattice. Computes L1/L2 distance to MIN_LATT.

    Args:
        l1 (float): L1 regularization factor.
        l2 (float): L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        """Compute L1L2Lattice regularization.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Regularization term.
        """
        regularization = 0.0
        if self.l1:
            regularization += self.l1 * K.sum(K.abs(x - MIN_LATT))
        if self.l2:
            regularization += self.l2 * K.sum(K.square(x - MIN_LATT))
        return regularization

    def get_config(self):
        """Get configuration of the regularizer.

        Returns:
            dict: Regularizer configuration.
        """
        return {"l1": float(self.l1), "l2": float(self.l2)}


def l1lattice(l=0.01):
    """Alias for L1L2Lattice with L1 regularization.

    Args:
        l (float): L1 regularization factor.

    Returns:
        L1L2Lattice: L1L2Lattice regularizer instance.
    """
    return L1L2Lattice(l1=l)


def l2lattice(l=0.01):
    """Alias for L1L2Lattice with L2 regularization.

    Args:
        l (float): L2 regularization factor.

    Returns:
        L1L2Lattice: L1L2Lattice regularizer instance.
    """
    return L1L2Lattice(l2=l)


def l1_l2lattice(l1=0.01, l2=0.01):
    """Alias for L1L2Lattice with both L1 and L2 regularization.

    Args:
        l1 (float): L1 regularization factor.
        l2 (float): L2 regularization factor.

    Returns:
        L1L2Lattice: L1L2Lattice regularizer instance.
    """
    return L1L2Lattice(l1=l1, l2=l2)
