"""DAS"""

import os

# Keras 3 defaults to TensorFlow unless a backend is selected first.
# DAS ships against PyTorch in this repository, so keep that as the default.
os.environ.setdefault("KERAS_BACKEND", "torch")

__version__ = "0.32.8"
