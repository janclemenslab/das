import json

import pytest


@pytest.fixture
def tiny_whisper_speech_model(tmp_path):
    from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperTokenizer
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode

    tokens = list(bytes_to_unicode().values()) + [
        "<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>",
        *[f"<|{index * 0.02:.2f}|>" for index in range(1501)],
    ]
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    (directory / "vocab.json").write_text(json.dumps(dict(zip(tokens, range(len(tokens))))))
    (directory / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = WhisperTokenizer(
        str(directory / "vocab.json"), str(directory / "merges.txt"),
        language="english", task="transcribe", predict_timestamps=True,
        additional_special_tokens=tokens[257:261],
    )
    config = WhisperConfig(
        vocab_size=len(tokenizer), d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_source_positions=1500, max_target_positions=128,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id, decoder_start_token_id=257,
        suppress_tokens=[], begin_suppress_tokens=[], das_whisper_asr=True,
        total_spec_columns=3000, cluster_codebook={}, species_codebook={},
    )
    model = WhisperForConditionalGeneration(config)
    model.generation_config.no_timestamps_token_id = 260
    model.generation_config.lang_to_id = {"<|en|>": 258}
    model.generation_config.task_to_id = {"transcribe": 259}
    model.generation_config.alignment_heads = [[0, 0]]
    model.generation_config.language = "english"
    return model, tokenizer
