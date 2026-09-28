"""DAS"""

__version__ = "1.0a2"

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def train(*args, **kwargs):
    from .api import train as _train

    return _train(*args, **kwargs)


def predict(*args, **kwargs):
    from .api import predict as _predict

    return _predict(*args, **kwargs)
