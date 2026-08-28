"""DAS"""

import os
import sys

# Keras 3 defaults to TensorFlow unless a backend is selected first.
# DAS ships against PyTorch in this repository, so keep that as the default.
os.environ.setdefault("KERAS_BACKEND", "torch")

__version__ = "0.33.0"

# Preserve the import paths exposed by the TensorFlow-backed releases.
from . import io, models, npy_dir

data = io
data_hash = io.data_hash
kapre = models.kapre
menagerie = models.menagerie
tcn = models.tcn_layers

for _name, _module in {
    "data": data,
    "data_hash": data_hash,
    "kapre": kapre,
    "menagerie": menagerie,
    "tcn": tcn,
}.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)

for _name in ("augmentation", "backend", "backend_keras", "filterbank", "time_frequency", "utils"):
    sys.modules.setdefault(f"{__name__}.kapre.{_name}", getattr(kapre, _name))
for _name in ("tcn", "tcn_new"):
    sys.modules.setdefault(f"{__name__}.tcn.{_name}", sys.modules[f"{tcn.__name__}.{_name}"])

del _name, _module
