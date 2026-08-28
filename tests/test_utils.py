from unittest.mock import Mock, call

from das import utils
from das.models import loading


def test_load_model_from_params_retries_loose_with_fresh_model(monkeypatch, caplog):
    strict_model = Mock()
    strict_model.load_weights.side_effect = ValueError("mismatch")
    legacy_model = Mock()
    loose_model = Mock()
    make_model = Mock(side_effect=[strict_model, legacy_model, loose_model])
    monkeypatch.setattr(utils, "load_params", lambda *_args, **_kwargs: {"model_name": "tcn"})
    monkeypatch.setattr(utils, "_download_if_url", lambda filename: filename)
    monkeypatch.setattr(loading, "_load_legacy_h5_weights", Mock(side_effect=ValueError("legacy mismatch")))

    model = utils.load_model_from_params("example", {"tcn": make_model}, compile=False)

    assert model is loose_model
    assert strict_model.load_weights.call_args_list == [call("example_model.h5", skip_mismatch=False, by_name=False)]
    assert loose_model.load_weights.call_args_list == [call("example_model.h5", skip_mismatch=True, by_name=True)]
    assert "Strict weight loading failed" in caplog.text
