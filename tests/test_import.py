import pytest


def test_import():
    from das import annot
    from das import augmentation
    from das import block_stratify
    from das import cli
    from das import evaluate
    from das import event_utils
    from das import io
    from das import make_dataset
    from das import models
    from das import tracking
    from das import postprocessing
    from das import predict
    from das import pulse_utils
    from das import segment_utils
    from das import train
    from das import utils
    from das.io import data_hash
    from das.io import npy_dir
    import das.models.kapre
    import das.models.menagerie
    import das.models.tcn


def test_model_registry():
    from das import models

    assert "tcn" in models.model_dict
    assert "tcn_stft" in models.model_dict
    assert callable(models.load_model_and_params)
