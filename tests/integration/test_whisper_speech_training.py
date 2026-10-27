import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from das import api
from das.whisper_asr import WhisperTranscriber
from das.whisperseg import train as training
from das.whisperseg.model import load_inference_model


@pytest.mark.parametrize("method", ["lora", "full"])
def test_speech_fit_adds_labels_and_exports_a_transcription_checkpoint(tmp_path, monkeypatch, tiny_whisper_speech_model, method):
    torch.set_num_threads(2)
    model, tokenizer = tiny_whisper_speech_model
    encoder_weights = {name: p.detach().clone() for name, p in model.model.encoder.named_parameters()}
    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "speech.wav", np.zeros(16000, dtype=np.float32), 16000)
    pd.DataFrame({"name": ["custom label"], "start_seconds": [0.1], "stop_seconds": [0.8]}).to_csv(
        data / "speech_annotations.csv", index=False,
    )
    monkeypatch.setattr(training, "load_model", lambda **_: (model, tokenizer))
    gradients = []

    def check_gradients(module, optimizer):
        parameters = [(name, value) for name, value in module.model.named_parameters() if value.requires_grad]
        assert all(value.grad is None or torch.isfinite(value.grad).all() for _, value in parameters)
        if method == "lora":
            label_parameters = [value for name, value in parameters if "trainable_tokens_delta" in name]
            assert label_parameters and all(value.grad is not None and value.grad.abs().sum() > 0 for value in label_parameters)
        base = module.model.get_base_model() if method == "lora" else module.model
        encoder = base.model.encoder
        assert encoder.embed_positions.weight.requires_grad is False
        assert any(p.grad is not None for p in encoder.parameters()) == (module.current_epoch >= 1)
        assert any(p.grad is not None for p in base.model.decoder.parameters())
        if method == "lora":
            assert all(not p.requires_grad for name, p in base.named_parameters()
                       if "lora_" not in name and "trainable_tokens_delta" not in name)
        gradients.append(True)

    monkeypatch.setattr(training.WhisperSegLightningModule, "on_before_optimizer_step", check_gradients)
    output = training.train(
        str(tmp_path / "run"), str(data), initial_model_path="whisper:large", training_method=method,
        device="cpu", num_epochs=1, validation_fraction=0, test_fraction=0, batch_size=1,
        accumulate_grad_batches=1, max_num_steps_per_epoch=1, max_length=32,
        early_stopping=False, reduce_lr=False,
        freeze_encoder=True, unfreeze_encoder_after=1,
        save_resumable_checkpoint=True,
    )

    assert gradients
    assert all(torch.equal(p, encoder_weights[name.replace(".base_layer", "")].to(p))
               for name, p in model.model.encoder.named_parameters() if "lora_" not in name)
    loaded, loaded_tokenizer = load_inference_model(output)
    assert loaded.config.das_whisper_asr
    assert loaded_tokenizer.encode("custom label", add_special_tokens=False) == loaded.config.das_annotation_token_ids
    assert api._is_whisper_asr_predict_source(output)
    assert api._is_whisperseg_predict_source(output)
    checkpoint = next((tmp_path / "run/checkpoints").glob("*.ckpt"))
    resumed = training._load_training_checkpoint(checkpoint)
    assert resumed.model.config.das_whisper_asr
    assert resumed.tokenizer.get_vocab() == loaded_tokenizer.get_vocab()
    transcriber = WhisperTranscriber(str(output), device="cpu")
    transcriber.pipeline.generation_config.max_new_tokens = 12
    predictions = transcriber.transcribe(np.zeros(16000, dtype=np.float32), num_beams=1, length_penalty=1.0)
    assert predictions.columns.tolist() == ["name", "start_seconds", "stop_seconds"]
    training.train(
        str(tmp_path / "resumed"), str(data), initial_model_path=str(checkpoint),
        device="cpu", num_epochs=2, early_stopping=False,
        save_resumable_checkpoint=True,
    )
    resumed_checkpoint = next((tmp_path / "resumed/checkpoints").glob("*.ckpt"))
    assert torch.load(resumed_checkpoint, map_location="cpu", weights_only=True)["global_step"] == 2
    continued = training._load_training_checkpoint(resumed_checkpoint)
    assert continued.hparams.unfreeze_encoder_after == 1
    if method == "full":
        assert any(not torch.equal(p, encoder_weights[name]) for name, p in continued.model.model.encoder.named_parameters())
    else:
        encoder = continued.model.get_base_model().model.encoder
        assert any(torch.count_nonzero(p) for name, p in encoder.named_parameters() if "lora_B" in name)
        assert all(torch.equal(p, encoder_weights[name.replace(".base_layer", "")].to(p))
                   for name, p in encoder.named_parameters() if "lora_" not in name)
