import torch

from das.whisper_asr import add_label_vocabulary
from das.whisperseg.model import _model_from_payload, checkpoint_payload


def test_added_labels_preserve_speech_vocabulary_weights_and_timestamp_range(tiny_whisper_speech_model):
    model, tokenizer = tiny_whisper_speech_model
    original_vocab = tokenizer.get_vocab()
    original_weights = model.get_input_embeddings().weight.detach().clone()
    boundary = tokenizer.eos_token_id
    labels = ["pulse", "custom label", "mouse 🐭"]

    tokenizer = add_label_vocabulary(model, tokenizer, labels)

    for token, index in original_vocab.items():
        new_index = tokenizer.convert_tokens_to_ids(token)
        assert new_index == index + (len(labels) if index >= boundary else 0)
        torch.testing.assert_close(model.get_input_embeddings().weight[new_index], original_weights[index])
    for index, label in enumerate(labels):
        assert tokenizer.encode(label, add_special_tokens=False) == [boundary + index]
        assert tokenizer.decode([boundary + index]) == label
    assert model.config.eos_token_id == tokenizer.eos_token_id
    assert model.get_input_embeddings().padding_idx == tokenizer.pad_token_id
    assert model.generation_config.no_timestamps_token_id + 1 == tokenizer.convert_tokens_to_ids("<|0.00|>")
    assert model.generation_config.lang_to_id["<|en|>"] == tokenizer.convert_tokens_to_ids("<|en|>")
    assert model.generation_config.task_to_id["transcribe"] == tokenizer.convert_tokens_to_ids("<|transcribe|>")
    restored, restored_tokenizer = _model_from_payload(checkpoint_payload(model, tokenizer, 0))
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    torch.testing.assert_close(restored.get_input_embeddings().weight, model.get_input_embeddings().weight)
