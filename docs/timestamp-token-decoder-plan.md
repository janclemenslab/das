# Native timestamp-token decoder for DAS

## Goal

Add `decoder_type: timestamp` as a small autoregressive Transformer decoder for native DAS encoders, including Conformer. It should generate timestamped annotations directly while reusing DAS's existing frontend/encoder pretraining and checkpoint paths.

The token grammar is deliberately compact:

```text
<BOS> <T_onset> <CLASS> <T_offset> ... <EOS>
```

Timestamp resolution is the frontend hop (typically 4 ms). Do not add a tokenizer package or another dependency.

## Locked decisions

- Use grammar-constrained greedy decoding, not beam search.
- Preserve exact CSV/JSON intervals, including nested and same-class overlaps.
- Support CLI, Python API, checkpoints, and the training GUI.
- Generated boundaries are authoritative. Do not apply gap filling, minimum-duration filtering, or postprocessing tuning.
- Reject legacy `npy_dir` training for this decoder because dense labels cannot preserve exact interval multiplicity.
- Use the decoder with native DAS frontends/encoders; WhisperSeg remains a separate backend.

## Implementation

### Configuration and model

- Add `timestamp` to the decoder config types and GUI selector. Reuse `decoder_num_heads`, `decoder_num_layers`, `decoder_dropout`, and `max_length`; no new tuning fields are needed.
- Require `max_length >= 5` and a known `num_time_steps`. The default `max_length=100` permits `(100 - 2) // 3 = 32` annotations per chunk.
- Assign token IDs as follows, where `F` is the maximum frontend-frame count and class indices exclude noise:
  - `PAD=0`, `BOS=1`, `EOS=2`
  - `T_k = 3 + k` for `k in [0, F]`
  - `CLASS_c = 3 + (F + 1) + (c - 1)` for DAS class index `c >= 1`
- Build the decoder from PyTorch's `TransformerDecoder` with causal target attention, encoder and target positional embeddings, encoder padding masks, `dim_feedforward=4 * encoder_dim`, and a tied token embedding/output projection.
- Under teacher forcing, `DASModel.forward(..., decoder_input_ids=...)` returns token logits. Without decoder inputs it returns greedy-generated token IDs and lengths. Native dense decoders retain their current contract.
- Train timestamp models with padding-ignored token cross-entropy and log token accuracy. Do not apply the framewise syllable-count loss or `cross_entropy_weight`.
- Reuse `initial_model`, supervised checkpoint weight loading, and encoder freezing.

### Targets and overlap preservation

- Add a timestamp target mode to `AudioDirGenerator`/`AudioDirDataModule`. It returns a fixed-size integer tensor of `[onset_bin, class_index, offset_bin]` triples plus the number of valid triples.
- Build triples directly from annotation rows rather than dense frame labels. Clip annotations to the chunk, round both boundaries to the nearest frontend frame, retain point events at one timestamp, and enforce at least one frame for segment classes.
- Sort triples deterministically by onset, offset, and class. This permits overlapping and nested intervals because only onsets—not prior offsets—must be nondecreasing.
- Raise a clear error if a chunk contains more than `(max_length - 2) // 3` annotations. Never truncate labels silently.
- Reject `decoder_type=timestamp` when the training source is `npy_dir`.

### Constrained generation

- Allow `EOS` only between complete triples.
- Constrain onset tokens to valid encoder frames and to be nondecreasing.
- Allow only non-noise class tokens in the class position.
- For segment classes require `offset > onset`; for event classes require `offset == onset`.
- Stop at `EOS` or `max_length`. Greedy decoding may recompute the short prefix each step; add KV caching only if profiling later shows it matters.

### Prediction, chunking, and evaluation

- Have `predict_step` return structured decoded triples, not probability traces. Extend `prediction_results` to recognize this output and produce the existing `name`, `start_seconds`, `stop_seconds` DataFrames/CSV files.
- Deduplicate overlapping chunks by ownership: chunk `i` owns absolute onsets in `(start_i, start_{i+1}]`; the first chunk also owns time zero and the last chunk owns through EOF. This rejects zero-onset continuation copies from later overlapping chunks without merging legitimate overlaps.
- Validate that every segment duration is no longer than the chunk overlap, `num_time_steps - chunk_stride`, so the owning chunk contains the complete interval. Tell users to increase chunk size or overlap when this fails.
- Keep `das.predict`, raw-audio prediction, output suffixes, merging, and checkpoint loading behavior unchanged from the caller's perspective.
- For evaluation, rasterize direct intervals only to populate the existing dense confusion report and reuse the existing interval matcher for segment/event metrics. Explicit postprocessing tuning should raise an unsupported-operation error for timestamp models; automatic training-time tuning should be skipped.

### GUI

- Add a timestamp decoder form with heads, layers, dropout, and maximum token length.
- Persist these values through GUI state/YAML round trips.
- The prediction GUI needs no timestamp-specific controls because greedy generation uses checkpoint settings.

## Verification

- Config, CLI, GUI selection, YAML round trips, and checkpoint reconstruction.
- Token IDs, interval-to-token encoding, target capacity errors, timestamp quantization, point events, nested intervals, and same-class overlaps.
- Teacher-forced loss, token accuracy, causal/padding masks, variable input lengths, and grammar-constrained greedy generation.
- Direct raw-audio and file prediction, chunk ownership/deduplication, CSV output, checkpoint reload, and interval-based evaluation.
- Focused model, audio-directory, prediction, configuration, and GUI tests, followed by the full suite in the `das-conformer` conda environment.

## Current baseline

Before implementation, the focused baseline produced 105 passing tests and one unrelated failure: `test_audio_dir_accepts_zarr_audio_with_sidecar_samplerate_and_dataset` uses `Group.create_dataset`, which is absent from the installed Zarr API. Do not fold that dependency mismatch into this feature.
