import pytest

def test_import():
    from dss import train
    from dss import predict
    from dss import evaluate
    from dss import utils
    from dss import utils_plot
    from dss import data
    from dss import io
    from dss import npy_dir
    from dss import models
    from dss import pulse_utils
    from dss import event_utils
    from dss import segment_utils
    import dss.kapre
    import dss.tcn


