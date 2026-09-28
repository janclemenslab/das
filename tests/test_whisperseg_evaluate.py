import json

import numpy as np

from das.whisperseg.evaluate import evaluate


class FakeSegmenter:
    def segment(self, audio, **kwargs):
        return {}

    def segment_score(self, prediction, label, target_cluster=None):
        return np.int64(1), np.int64(2), np.int64(4)

    def frame_score(self, prediction, label, target_cluster=None):
        return np.int64(3), np.int64(5), np.int64(6)


def test_evaluate_returns_json_serializable_metrics():
    result = evaluate(
        audio_list=[np.zeros(10)],
        label_list=[{"sr": 1000}],
        segmenter=FakeSegmenter(),
        batch_size=1,
        max_length=8,
        num_trials=1,
    )

    json.dumps(result)
    assert result["segment_wise"][:3] == [1, 2, 4]
    assert result["frame_wise"][:3] == [3, 5, 6]
