from types import SimpleNamespace

import numpy as np

from das.whisperseg.datautils import VocalSegDataset


def test_validation_crops_repeat_without_consuming_training_rng(monkeypatch):
    class Features:
        n_fft = 4

        def __call__(self, audio, **kwargs):
            return {"input_features": [audio[None, :]]}

    monkeypatch.setattr(VocalSegDataset, "get_feature_extractor_bank", lambda *args: {"1000-0.001-0-None": Features()})
    monkeypatch.setattr(VocalSegDataset, "_read_record_audio", lambda *args: np.arange(200, dtype=np.float32))
    label = dict(sr=1000, spec_time_step=0.001, min_frequency=0, species="unknown",
                 onset=np.array([0.08]), offset=np.array([0.12]), cluster_id=np.array([0]))
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text, **kwargs: [1, *map(ord, text), 2])
    records = [SimpleNamespace(label=label)] * 2
    validation = VocalSegDataset(records, tokenizer, 128, 100, {"unknown": "<|unknown|>"}, deterministic=True)
    training = VocalSegDataset(records, tokenizer, 128, 100, {"unknown": "<|unknown|>"})
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected_training = training[0]
        np.random.seed(123)
        first = validation[0]
        validation[1]
        second = validation[0]
        actual_training = training[0]
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            np.testing.assert_array_equal(actual_training[key], expected_training[key])
        assert not np.array_equal(first["input_features"], validation[1]["input_features"])
        assert not np.array_equal(actual_training["input_features"], training[0]["input_features"])
    finally:
        np.random.set_state(state)
