from pathlib import Path

import huggingface_hub
import pytest

from das import cli
from das.config import Config
from das.whisperseg.catalog import MODEL_OPTIONS, MODEL_REPO, resolve_model_checkpoint


def test_published_models_parse_in_cli_and_download_on_use(monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "/cache/" + kwargs["filename"]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert set(MODEL_OPTIONS) == {
        "whisperseg-aer:tiny-ms", "whisperseg-aer:small-ms",
        "whisperseg-aer:base-animal-vad", "whisperseg-aer:animal-vad", "whisperseg-aer:large-ms",
    }
    for index, (alias, (_, filename)) in enumerate(MODEL_OPTIONS.items()):
        train = cli.parse_command_config(
            ["train", "--data-dir", "data", "--encoder", "whisperseg", "--initial-model", alias]
        )
        predict = cli.parse_command_config(["predict", "--data-dir", "data", "--checkpoint", alias])
        assert train.initial_model == alias
        assert predict.checkpoint == alias
        assert len(calls) == index
        assert resolve_model_checkpoint(alias) == "/cache/" + filename
        assert calls[-1] == {"repo_id": MODEL_REPO, "filename": filename}
    assert resolve_model_checkpoint(str(Path("/tmp/local.ckpt"))) == "/tmp/local.ckpt"

    with pytest.raises(ValueError, match="Unknown WhisperSeg AER model"):
        Config(mode="predict", data_dir="data", checkpoint="whisperseg-aer:unknown").validate()
