# WhisperSeg training and prediction

## Starting models and fine-tuning

The model selectors offer the compact FP16, reduced-vocabulary WhisperSeg AER
checkpoints: Tiny multispecies (`whisperseg-aer:tiny-ms`), Base Animal VAD
(`whisperseg-aer:base-animal-vad`), Small multispecies (`whisperseg-aer:small-ms`),
Large Animal VAD (`whisperseg-aer:animal-vad`) and Large multispecies
(`whisperseg-aer:large-ms`). Original WhisperSeg and pruned models are not offered.
Models download on first use. Prediction preserves their FP16 weights.

## Human speech transcription with timestamps

For spoken words, select **Whisper Large · speech transcription · FP16** in the
prediction checkpoint selector, or run:

```shell
das predict --data-dir /path/to/speech.wav --checkpoint whisper:large
```

This uses the original [OpenAI Whisper Large](https://huggingface.co/openai/whisper-large)
with its full vocabulary and FP16 weights. Language is detected automatically;
words are transcribed in the source language. Each word, including punctuation,
is saved as a normal DAS annotation with `name`, `start_seconds` and `stop_seconds`,
so timestamps and words can be viewed and edited in the GUI. Prediction also
works on the current audio selection; its annotations use the recording timeline.

Whisper handles recordings longer than 30 seconds using its own long-form
decoding. WhisperSeg's frequency scaling, event postprocessing and short decoder
token limit do not apply. Speech transcription is prediction-only; segmentation
evaluation is disabled. The model downloads on first use.

DAS corrects the token-boundary convention used by Transformers 4.57's word
decoder: each token spans its own onset to the next token's onset. This avoids
shifting word times one token earlier and producing zero-length first words.
For long recordings, decoding advances to the last aligned spoken token, ignoring
trailing punctuation, as in OpenAI Whisper's word-timestamp decoding. This avoids
replaying a phrase when a predicted segment timestamp falls before its final words.
If a word's estimated end overlaps the next word, DAS trims its end to the next
word's onset. Spoken repetitions are retained.
Word boundaries remain estimates from Whisper's attention alignment.

## WhisperSeg fine-tuning

WhisperSeg fine-tuning defaults to **decoder LoRA**, using the selected adaptation
recipe from our experiments:

- Frozen FP16 backbone; FP32 adapters and AdamW state.
- Rank 16, alpha 32, no adapter dropout or bias adaptation. Adapters cover decoder
  self- and cross-attention q/k/v/out projections and both feed-forward layers.
- Learning rate `1e-4`, AdamW epsilon `1e-4`, weight decay `0.01`, gradient norm clip 1.
- Microbatch 1, accumulation 4, target length 128, seed 0, validation fraction 0.125.
- Non-reentrant decoder gradient checkpointing and no generation cache during fitting.
- Validation-loss plateau reduction (patience 2, factor 0.5), early stopping
  (patience 5), at most 24 epochs; no linear warmup schedule.

```shell
das train --data-dir /path/to/annotated-audio --encoder whisperseg \
  --initial-model whisperseg-aer:small-ms --output-dir /path/to/run
```

Choose `full` in the GUI or add `--whisperseg-training-method full` to use ordinary
FP32 fine-tuning with the previous full-training defaults. `--freeze-encoder`
restricts full training to the decoder. `--accumulate-grad-batches` sets accumulation.
Explicit training settings override defaults.

LoRA training checkpoints live under `checkpoints/` and retain the base, adapter
and optimizer state. The best model is also exported into the run folder as a
merged FP16 DAS `.ckpt`, usable for prediction or another fine-tuning run without
an adapter loader. Merging retains the original inference architecture and size.
Select a training checkpoint in the GUI's **Initial model** field or pass it as
`--initial-model` in the CLI to resume automatically. DAS restores the existing
adapters, optimizer moments, learning-rate scheduler, epoch/step counters and
saved training settings, including preprocessing and class IDs. New training
checkpoints embed the tokenizer and backbone, so the original downloaded model
is not needed for resuming. A merged prediction checkpoint starts a fresh
fine-tuning run instead.

`num_epochs` remains the total epoch limit, including epochs already completed.
Increase it beyond the saved epoch when continuing a completed run. Use the same
training data to preserve the original split. Resume preserves training state;
it does not promise an identical random crop or shuffle sequence.

The selected recipe was developed on Large; quality on other sizes and new
recordings should be checked against representative held-out annotations.

## Audio windows and syllable boundaries

WhisperSeg splits each input recording into model windows. Their duration is
`total_spec_columns * spec_time_step`: normally 1,000 spectrogram columns at
0.0025 seconds per column, giving a 2.5-second window before any frequency/time
scaling. This windowing is handled inside WhisperSeg.

A syllable can cross a window boundary. Training targets clip such syllables to
the visible part of each window. During prediction, WhisperSeg joins the last
event of one window to the first event of the next when their class matches and
the preceding offset exactly equals the following onset. A predicted gap or
class mismatch can still leave fragments.

DAS defaults to one prediction pass (`num_trials=1`). Set `--num-trials 3` for
three shifted passes that place window boundaries at different times. Matching
predictions are consolidated across passes, reducing boundary artifacts at the
cost of repeating inference. Separate files and prediction calls are processed
independently; events are not joined across them.

## Decoder token budget

`generation_max_length` limits the **decoder sequence length for each model
window in each pass**, including the decoder prefix. It is not an event count,
an audio-duration limit, or a shared budget for the recording or batch.

An event is encoded as an onset timestamp, a numeric class ID and an offset
timestamp. Each timestamp occupies one token; the class ID can occupy one or
more tokens. Prefix, species and end-of-sequence tokens also consume space.
With the usual three-token decoder prefix, one species token, an end token and
a one-token class ID for every event, the budget is approximately:

| Setting | Decoder tokens per window | Complete events per window |
| --- | ---: | ---: |
| DAS default | 128 | 41 |
| Original WhisperSeg default | 448 | 147 |

Multi-token class IDs or other generated tokens reduce the event capacity. At
2.5 seconds per window, 41 events correspond to roughly 16 events per second
within that window. Dense bursts can exhaust the budget even when the average
event rate over the recording is low.

Generation can stop earlier when the model emits an end token. If it reaches
the cap instead, later events may be omitted, and an incomplete final event is
discarded by the event parser. DAS does not automatically retry that window
with a larger budget. More beams or passes do not increase the token budget of
an individual window.

For dense recordings, inspect predictions for truncation and increase the cap
when needed, for example:

```shell
das predict --data-dir /path/to/wavs --checkpoint /path/to/model.ckpt --generation-max-length 448
```

The larger cap permits longer output; it does not force every window to use all
448 tokens. Longer decoding can increase runtime and decoder-cache memory.
The separate `max_length` setting controls training target length; changing
`generation_max_length` does not change that training limit.
