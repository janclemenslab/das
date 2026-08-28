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


def test_legacy_imports():
    from das import data
    from das import data_hash
    from das import menagerie
    from das import npy_dir
    import das.kapre
    import das.kapre.augmentation
    import das.kapre.time_frequency
    import das.loss
    import das.models_legacy
    import das.spec_utils
    import das.tcn
    import das.tcn.tcn
    import das.tcn.tcn_new
    import das.utils_plot


def test_legacy_loader_api_and_train_args():
    import inspect

    from das import utils
    from das.train import train

    assert callable(utils.load_model)
    assert callable(utils.load_model_from_params)
    assert callable(utils.load_model_and_params)

    parameters = inspect.signature(train).parameters
    for name in ("resnet_compute", "resnet_train", "tmse_weight"):
        assert name in parameters


def test_model_registry():
    from das import models

    assert "tcn" in models.model_dict
    assert "tcn_stft" in models.model_dict
    assert "stft_res_dense" in models.model_dict
    assert callable(models.tcn)
    assert callable(models.load_model_and_params)
