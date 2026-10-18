"""Exercise production fitting, checkpoint restore and merged FP16 prediction."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperTokenizer

from das.whisperseg import model as modeling
from das.whisperseg import train as training
from das import api
from das.cli import parse_command_config
from das.gui_helpers import train_config_to_gui_state, train_gui_state_to_config


@pytest.mark.parametrize("method,device", [("lora", "cpu"), ("lora", "mps"), ("full", "cpu")])
def test_fit_restore_and_predict(tmp_path, monkeypatch, method, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    tokens = [str(i) for i in range(10)] + [
        "<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|notimestamps|>",
        "<|unknown|>", *[f"<|{i}|>" for i in range(17)],
    ]
    (tmp_path / "vocab.json").write_text(json.dumps(dict(zip(tokens, range(len(tokens))))))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = WhisperTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"),
                                 language="english", additional_special_tokens=tokens[10:])
    config = WhisperConfig(
        vocab_size=len(tokenizer), d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_source_positions=8, max_target_positions=16, pad_token_id=10, bos_token_id=10,
        eos_token_id=10, decoder_start_token_id=11, suppress_tokens=[], begin_suppress_tokens=[],
        total_spec_columns=16, species_codebook={"unknown": "<|unknown|>"}, cluster_codebook={},
    )
    source = WhisperForConditionalGeneration(config).half()
    original_weights = {n: p.clone() for n, p in source.named_parameters()}
    original_encoder = {n: p.clone() for n, p in source.model.encoder.named_parameters()}
    original_decoder = {n: p.clone() for n, p in source.model.decoder.named_parameters()}
    source_path = tmp_path / "source.ckpt"
    torch.save({"whisperseg": modeling.checkpoint_payload(source, tokenizer, 0)}, source_path)
    batch = {"input_features": torch.randn(80, 16), "labels": torch.tensor([14, 15, 0, 18, 10])}
    monkeypatch.setattr(training, "get_audio_and_label_paths", lambda *a, **k: (["clip.wav"], ["clip.csv"]))
    monkeypatch.setattr(training, "determine_default_config", lambda *a, **k: {"spec_time_step": 0.0025})
    monkeypatch.setattr(training, "get_cluster_codebook", lambda *a, **k: {"custom label": 0, "another": 12})
    monkeypatch.setattr(training, "build_vocalseg_clip_records", lambda *a, **k: ([batch] * 5, [batch], []))
    monkeypatch.setattr(training, "VocalSegDataset", lambda records, *a, **k: records)
    run = tmp_path / "run"
    run.mkdir()
    updates = []

    def check_gradients(module, optimizer):
        gradients = [p.grad for p in module.model.parameters() if p.grad is not None]
        assert gradients and all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients)
        assert optimizer.defaults["eps"] == (1e-4 if method == "lora" else 1e-8)
        updates.append(module.global_step)

    monkeypatch.setattr(training.WhisperSegLightningModule, "on_before_optimizer_step", check_gradients)
    exported = Path(training.train(
        str(run), str(tmp_path), initial_model_path=str(source_path), training_method=method,
        device=device, total_spec_columns=16, num_epochs=1, max_length=16,
        batch_size=1, accumulate_grad_batches=2, freeze_encoder=True,
        linear_lr_schedule=method == "full", reduce_lr=method != "full", warmup_steps=1,
    ))
    best = next((run / "checkpoints").glob("*.ckpt"))
    checkpoint = modeling._torch_load(best)
    assert checkpoint["global_step"] == 3  # Five microbatches, including the final partial update.
    assert updates == [0, 1, 2]
    moments = checkpoint["optimizer_states"][0]["state"].values()
    assert moments and all(state["exp_avg"].dtype == torch.float32 for state in moments)
    restored = training.WhisperSegLightningModule.load_from_checkpoint(best, map_location="cpu", weights_only=True)
    assert restored.model.config.cluster_codebook == {"custom label": 0, "another": 12}
    if method == "lora":
        assert exported.parent == run and exported != best
        assert "whisperseg" not in checkpoint
        assert restored.model.get_base_model().model.decoder.is_gradient_checkpointing
        frozen = {n: p for n, p in restored.model.named_parameters() if not p.requires_grad}
        adapters = {n: p for n, p in restored.model.named_parameters() if p.requires_grad}
        assert all(p.dtype == torch.float16 for p in frozen.values())
        assert all(torch.equal(p, original_weights[n.removeprefix("base_model.model.").replace(".base_layer", "")])
                   for n, p in frozen.items())
        assert adapters and all("decoder" in n and "lora_" in n and p.dtype == torch.float32
                                for n, p in adapters.items())
        assert any("fc1" in n for n in adapters) and any("k_proj" in n for n in adapters)
        assert any("lora_B" in n and torch.count_nonzero(p) for n, p in adapters.items())
        expected = restored.model.merge_and_unload(safe_merge=True).half()
    else:
        expected = restored.model
        assert any(not torch.equal(p.half(), original_decoder[n]) for n, p in expected.model.decoder.named_parameters())
    loaded, _ = modeling.load_inference_model(exported)
    assert not loaded.training
    assert all(torch.equal(p.half(), original_encoder[n]) for n, p in loaded.model.encoder.named_parameters())
    assert not any("lora_" in n for n in loaded.state_dict())
    assert loaded.config.cluster_codebook == {"custom label": 0, "another": 12}
    assert all(torch.equal(p, expected.state_dict()[n]) for n, p in loaded.state_dict().items())
    if method == "lora":
        assert next(loaded.parameters()).dtype == torch.float16 and loaded.config.use_cache
    for segmenter in (modeling.WhisperSegmenter(str(exported), device=device),
                      modeling.WhisperSegmenterForEval(model_path=str(exported), device=device)):
        prediction = segmenter.segment(np.zeros(640, dtype=np.float32), 16000,
                                       num_trials=1, num_beams=1, max_length=12, batch_size=1)
        assert set(prediction) == {"onset", "offset", "cluster"}

    # Give the saved scheduler and learning rate values distinct from fresh defaults.
    for group in checkpoint["optimizer_states"][0]["param_groups"]:
        group["lr"] = 2e-5
    checkpoint["lr_schedulers"][0]["_last_lr"] = [2e-5, 2e-5]
    if method == "lora":
        checkpoint["lr_schedulers"][0]["num_bad_epochs"] = 1
    portable = tmp_path / "portable.ckpt"
    torch.save(checkpoint, portable)
    original_checkpoint = portable.read_bytes()
    source_path.unlink()
    resumed_steps = []

    def check_resume(module):
        assert module.global_step == 3 and module.current_epoch == 1
        assert module.training_method == method
        assert module.trainer.accumulate_grad_batches == 2
        assert module.total_training_steps == 3
        for name, value in module.state_dict().items():
            assert torch.equal(value.cpu(), checkpoint["state_dict"][name])
        optimizer = module.trainer.optimizers[0].state_dict()
        assert optimizer["param_groups"] == checkpoint["optimizer_states"][0]["param_groups"]
        for index, state in optimizer["state"].items():
            for name, value in state.items():
                assert torch.equal(value.cpu(), checkpoint["optimizer_states"][0]["state"][index][name])
        assert module.lr_schedulers().state_dict() == checkpoint["lr_schedulers"][0]
        resumed_steps.append(module.global_step)

    monkeypatch.setattr(training.WhisperSegLightningModule, "on_train_start", check_resume)
    monkeypatch.setattr(training, "determine_default_config", lambda *a, **k: pytest.fail("Rebuilt preprocessing"))
    monkeypatch.setattr(training, "get_cluster_codebook", lambda *a, **k: pytest.fail("Rebuilt class IDs"))
    resume_dir = tmp_path / "resumed"
    config = parse_command_config([
        "train", "--data-dir", str(tmp_path), "--output-dir", str(resume_dir),
        "--encoder", "whisperseg", "--initial-model", str(portable),
        "--num-epochs", "2", "--accelerator", device,
    ])
    if device == "mps":
        config = train_gui_state_to_config(train_config_to_gui_state(config))
    resumed_export = Path(api.train(config))
    continued = modeling._torch_load(next((resume_dir / "checkpoints").glob("*.ckpt")))
    assert resumed_steps == [3] and continued["global_step"] == 6
    assert updates == [0, 1, 2, 3, 4, 5]
    assert portable.read_bytes() == original_checkpoint
    assert resumed_export.is_file()
    if method == "lora":
        assert "whisperseg_training" in continued
        assert resumed_export.parent == resume_dir
