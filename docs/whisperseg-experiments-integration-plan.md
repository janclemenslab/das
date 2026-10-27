# WhisperSeg experiment review and DAS integration plan

Reviewed 16 October 2026. Scope: the local WhisperSeg inference, vocabulary,
pruning, Tiny/Small, training-memory, precision, LoRA and distillation experiments,
plus the nightingale encoder and fly-song experiments relevant to DAS.

Integrate compact inference first, then the validated Large LoRA recipe. Keep all
decoder layers in the standard model; offer trained 4D/2D checkpoints as explicit
speed choices. Preserve ordinary FP32 decoder training. Treat Small as a promising
capacity option and Tiny as a small-footprint option. These recommendations concern
the WhisperSeg backend, not a replacement for DAS's native models.

## What the experiments establish

### Vocabulary reduction and FP16 inference

The strongest evidence is conversion of the **same fitted checkpoint**, without
retraining. Independently fitted compact models answer a different question.

| Fitted Base conversion | Multispecies macro F1 | Adapted within-bird F1 | Adapted across-bird F1 | Across-bird peak MPS MiB |
| --- | ---: | ---: | ---: | ---: |
| Full vocabulary, FP32 | 0.9706 | 0.9397 | 0.9053 | 1317.7 |
| Reduced vocabulary, FP32 | 0.9706 | 0.9397 | 0.9053 | 520.2 |
| Reduced vocabulary, FP16 | 0.9695 | 0.9397 | 0.9029 | 260.1 |

Vocabulary removal preserves measured F1 and cuts Base peak allocation by 60.5%.
Adding FP16 cuts the original peak by 80.3%, with a 0.242 percentage-point
across-bird F1 loss. Base weight storage falls from 275.0 to 175.9 to 87.9 MiB.
The peak figures include other allocations and must not be interpreted as weight
bytes. Base runtime is a single cold pass, so it is weak throughput evidence.

Paired Tiny/Small conversions also preserve F1 with vocabulary removal. FP16's
largest reported loss across their four headline tests is 0.287 percentage points
(Small within-bird adaptation). Adapted across-bird F1 is 0.8764 for Tiny both
before and after FP16, and 0.9311 → 0.9304 for Small.

Speed depends on hardware and output length. Paired Tiny T4 inference reduces
allocated GPU memory from 0.183 to 0.058 GiB and gives about 1.08× full-to-compact
speedup; FP16 alone is slightly slower than vocabulary-only FP32. Released compact
Base is also slightly slower than original Base on T4 (9.853 vs 9.280 seconds).
Do not promise universal FP16 acceleration.

Decision: use physically compact checkpoints, preserving their stored dtype.
Retain a compact FP32 option for accuracy-sensitive use. Masking unused logits
alone does not deliver the measured weight/matrix reduction. Model vocabulary
must include required prompt/species/timestamp tokens and decimal class digits.

Sources: [paired Base raw results](../runs/whisperseg-aer-adaptation-20260927/paper/verification/fitted-vocabulary-ablation/),
[Tiny/Small quality results](../runs/whisperseg-aer-adaptation-20260927/full-vocab-recovered-20261015/quality-summary.json),
[current report](../runs/whisperseg-aer-adaptation-20260927/paper/whisperseg-aer.tex).

### Fewer decoder layers and smaller backbones

| Model | Multispecies macro F1 | Adapted within-bird F1 | Adapted across-bird F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Base, paired compact FP16 6D | 0.9695 | 0.9397 | 0.9029 | Standard Base capacity, converted fitted weights |
| Base compact 4D | 0.9713 | 0.9415 | 0.8899 | Useful speed option, with an across-bird cost |
| Base compact 2D | 0.9714 | 0.8985 | 0.8784 | Faster, with a larger adaptation cost |
| Tiny, compact-trained | 0.9280 | 0.9165 | 0.8716 | Smallest footprint; lower in-domain accuracy |
| Small, compact-trained | 0.9589 | 0.9503 | 0.9358 | Strong candidate between Base and Large |
| Large Animal VAD, full FP32 | 0.9850 | 0.9616 | 0.9417 | Strong accuracy reference; expensive inference |

Base 4D/2D are independently fitted systems, so their differences include fitting,
not just removal of layers. Their measured M1 Pro adapted inference times are
0.298/0.244 seconds per audio second, versus 0.454 for full FP32 Base. Tiny/Small
resources were measured on M2 Max; do not rank cross-host speeds as model effects.

The older README/comparison table includes independently fitted Base 6D/1D rows
that the current report excludes because they used 1,612 rather than 3,121 training
recordings. Keep these distinct from paired conversion of the fitted original
Base. Do not promote the 1D model or claim a matched 6D ranking until exact released
checkpoint hashes and training manifests are reconciled. The released-model table
is useful for choosing those particular artifacts, not causal training comparisons.

Decision: select already trained pruned checkpoints; do not prune arbitrary user
models during prediction. Keep 4D and 2D optional. Add Tiny/Small only after their
DAS-format artifacts and provenance pass the release checks below. A universal
default backbone is not established by one adaptation task.

Sources: [current report](../runs/whisperseg-aer-adaptation-20260927/paper/whisperseg-aer.tex),
[released-checkpoint results and exclusions](../runs/whisperseg-aer-adaptation-20260927/pan-uol-results-20261014/RESULTS.md).

### FP32 LoRA training

The successful recipe means **FP32 trainable adapters and Adam state on a frozen
FP16 backbone**. It does not mean putting the entire Large model in FP32.

| Same-source Large adaptation | Within-bird F1 | Across-bird F1 | Peak training/validation MPS GiB | Median optimizer update s |
| --- | ---: | ---: | ---: | ---: |
| Decoder LoRA, rank 16 | 0.9719 | 0.9481 | 3.682 | 4.473 |
| Full decoder FP32 | 0.9683 | 0.9311 | 15.533 | 6.479 |
| Full decoder FP16 mixed | 0.9683 | 0.9308 | 16.578 | 7.422 |

LoRA uses 17,039,360 trainable parameters versus 841,571,840 for the full decoder,
76.3% less peak MPS memory and 1.45× faster updates than FP32. All F1 scores use
FP16 inference exports, with LoRA merged into the backbone. Merging leaves the full
Large inference architecture: LoRA's training savings do not make inference Tiny-sized.

Carry over the selected recipe as one preset:

- Freeze the backbone in FP16; train adapters and Adam moments in FP32.
- Rank 16, alpha 32, zero adapter dropout, no adapted biases.
- Target decoder self- and cross-attention q/k/v/out projections and fc1/fc2.
  Do not target encoder modules through an unrestricted name match.
- Learning rate 1e-4, AdamW epsilon 1e-4, weight decay 0.01, gradient norm clip 1.
- Decoder checkpointing with `use_reentrant=False`; disable generation cache
  during fitting and restore it for inference.
- Microbatch 1, accumulation 4, target cap 128, seed 0; no linear warmup schedule.
- Validation-loss plateau patience 2, factor 0.5; early-stopping patience 5;
  maximum 24 epochs. The experiment selected epoch 12 and stopped after 17.

The old rank-8 q/v recipe at learning rate 3e-6 reached only 0.8185 across-bird F1.
Rank, coverage and learning rate changed together; copy the validated recipe rather
than attributing the gain to one change. FP32 control also stopped after 17 epochs;
mixed precision was time-capped at 13, so it is not a convergence comparison.
The evidence is one seed and two held-out recordings on MPS. Validate CUDA and
other backbone sizes before recommending their LoRA training by default.

Sources: [verified comparison](../runs/whisperseg-aer-adaptation-20260927/large-lora-gap-20261015/comparison.json),
[selected recipe](../runs/whisperseg-aer-adaptation-20260927/large-lora-gap-20261015/recommended-lora.json),
[experimental implementation](../runs/whisperseg-aer-adaptation-20260927/large-fp16-lora-20261015/trial.py).
PEFT documents [FP32 adapter handling](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)
and [separate adapter and merged checkpoint formats](https://huggingface.co/docs/peft/en/developer_guides/checkpoint).

### Other results worth carrying forward

| Experiment | Finding | DAS decision |
| --- | --- | --- |
| FP32 training with compact vocabulary | Synthetic Base training driver peak 1.626 → 0.722 GiB; Large 18.916 → 17.119 GiB | Train compact models directly; vocabulary savings are useful before export too |
| Decoder checkpointing | Synthetic compact Large FP32 peak 17.120 → 15.688 GiB, with identical short-step losses and updated-weight hashes | Add a training option; enable it in the Large LoRA preset |
| MPS mixed precision | Higher driver peak in 11/12 short-step configurations; fitted Large also slower and larger than FP32 | Keep FP32 full-decoder training as the standard MPS recipe |
| Pure FP16 parameters and Adam | Non-finite loss at epoch 11 | Do not ship as a supported training preset |
| Disable autocast cache | Helps checkpointed mixed precision, but Large remains larger than FP32 checkpointing | Defer as a public setting; no general cache-tuning surface |
| Custom class names | Compact tokenizer encodes integer class IDs using retained digits | Keep normal DAS class-codebook behavior; test multiple and multi-digit IDs |
| Bounded generation | Fixed repeated decoding of one audio window | Preserve current per-window unique generation call and token cap in every inference/export path |
| Conformer distillation | Best students strong in-domain, but nightingale transfer F1 only about 0.13–0.24 | Defer general-purpose integration; a specialist research option |
| Frozen nightingale Animal2vec | Event F1 0.8841; BirdCODE 0.0301 and AVES2 0 in this syllable setup | Keep Animal2vec as a separate specialist route; coarse pooled embeddings are poor syllable defaults |
| Fly waveform specialists | High-resolution ConvResNet + TCN useful for pulses; long-context ConvResNet + Conformer useful for sine | Preserve existing native paths; do not route millisecond fly pulses through WhisperSeg by default |
| Fly activity hysteresis | Sine test interval F1 0.4417 → 0.5225 with validation-selected thresholds | Reuse existing postprocessing; address negative-channel false positives before stronger claims |
| Native timestamp decoder / SSL | Incomplete generated event sequences; existing SSL objective admits collapse | Keep experimental; no scaling or default promotion |

Training-step profiles use synthetic inputs and exclude whole-job loading/RAM.
MPS driver allocation and process RSS overlap and must not be added. Animal2vec's
split includes two channels of one recording across train/test; its independent
Clipper score is 0.8779. Fly file splits also share recording sessions.

Sources: [training-memory report sections](../runs/whisperseg-aer-adaptation-20260927/paper/whisperseg-aer.tex),
[distillation results](../runs/whisperseg-large-distillation-20260916/comparison.md),
[encoder comparison](../runs/nightingale-encoders-20260908/results.md),
[fly experiments](fly-waveform-conformer-experiment.md).

## What DAS already supports

- [Model loading](../src/das/whisperseg/model.py) reconstructs embedded configuration
  and weights without forcing FP32 during inference, and casts features to model
  dtype. Decoder depth and vocabulary size come from the checkpoint. Per-window
  bounded generation is already present. The separate training loader calls
  `model.float()` and repairs missing unknown-species tokens.
- The benchmark explicitly calls `.eval()`; production checkpoint inference does
  not. Fresh model construction leaves training mode enabled. Set evaluation mode
  at the inference boundary so dropout/layer dropping cannot make deployed behavior
  differ from the benchmark. Existing no-gradient segmentation is not a substitute.
- [The catalog](../src/das/whisperseg/catalog.py) and shared GUI/CLI selectors already
  expose seven Base/Large compact/pruned aliases. Tiny/Small are absent. Downloads
  are not pinned to a revision in this resolver.
- [Training](../src/das/whisperseg/train.py) supports encoder freezing and gradient
  accumulation internally. [Config](../src/das/config.py), [CLI](../src/das/cli.py),
  [API](../src/das/api.py) and GUI do not expose/forward accumulation. Training has
  no production LoRA/checkpointing mode, configurable Adam epsilon or gradient clip.
  PEFT is absent from [package dependencies](../pyproject.toml).
- The native checkpoint serializer assumes an ordinary Whisper model. A PEFT
  wrapper's state cannot simply be saved into that payload and loaded as native
  Whisper weights. The experimental trainer also monkeypatches production classes;
  those patches belong outside the integration.

## Implementation sequence and acceptance checks

### 1 Reconcile artifacts and finish compact inference support

Audit each candidate's checkpoint hash, training split, decoder depth, dtype,
tokenizer and generation configuration. Pin the catalog to a verified release
revision. Keep current aliases where correct; identify the legacy 6D/1D fits
clearly. Package trained Tiny/Small through the existing DAS checkpoint format,
then add their aliases to the existing selectors. Keep arbitrary model pruning
and conversion out of normal prediction.

First make production inference explicitly use evaluation mode in both segmenter
entry points. Check repeated predictions with a nonzero-dropout fixture and ensure
the training loader still returns a trainable model. Benchmark and production
must use the same decoding settings before resource comparisons guide defaults.

For user-trained models, extend the existing conversion module with the tested
vocabulary remapping only when needed for compact exports. Preserve class codebooks,
all required tokens, tied embeddings and every generation token-ID field. Write
a new output, leaving source weights intact. Offer FP32 or FP16 export; prediction
should use checkpoint dtype without another independent precision setting.

Touch: model/convert/catalog modules, catalog/model tests, packaging docs and GUI
selector tests. Verify native checkpoint round trips for all offered depths and
both dtypes; exact retained weights in FP32 conversion; normal/custom classes;
CPU/MPS/CUDA inference on available devices; no stalled window advancement or cap
violations. Use CPU FP32 if actual CPU validation requires it. Do not claim CUDA
accuracy from timing/event-count benchmarks. Reuse existing paired quality results.

### 2 Expose the training controls required by the successful recipes

Wire accumulation and decoder checkpointing through Config → CLI/GUI → API →
WhisperSeg trainer. Preserve the default full-decoder FP32 path. Count scheduler
steps using actual optimizer updates when accumulation is greater than one.
Disable cache only during training, restoring it for inference exports. Keep
recipe-specific epsilon and clipping with the recipe rather than adding a large
matrix of public precision/optimizer settings.

Verify configuration/GUI round trips, accumulation's actual optimizer-step count,
finite updates and frozen encoder weights. Reuse the short checkpointing
equivalence checks. These controls must also reach verbose post-training evaluation.

### 3 Integrate one LoRA adaptation mode

Add a WhisperSeg training-method choice, full decoder or LoRA, with the selected
Large recipe above as a preset. Resolve tokenizer additions and embedding resizing
before freezing/wrapping. Require the supported segmentation token inventory for
LoRA; ordinary class-name changes use the codebook and need no new token rows.
Filter optimizer groups to trainable parameters and verify their FP32 dtype.

Keep the validated preparation/export logic, replacing experimental monkeypatches
with direct production calls. Use PEFT as an optional training dependency, initially
testing the experiment's PEFT 0.18.0 with the pinned Transformers 4.57.1. Ordinary
prediction of a merged checkpoint must not require PEFT.

Keep LoRA training state/configuration distinct from the native inference payload.
Save the adapter recipe, exact base identity and optimizer/scheduler state needed
to restore training correctly. Export a separate merged FP16 ordinary DAS `.ckpt`
from the selected best checkpoint without altering the live resumable model.
Preserve tokenizer, class map, preprocessing and generation settings. Do not make
the generic inference loader understand unmerged adapters.

Touch: trainer, model export boundary, Config/API/CLI, GUI state/forms, training
preset, optional dependency metadata and focused tests. Verify real adapter updates,
unchanged frozen weights, finite FP32 gradients/Adam moments, training checkpoint
restore, best-checkpoint selection, merge/export/reload and custom-class prediction.
Port existing tiny-model tests instead of rerunning Large fits to test wiring.

### 4 Validate the integrated path before recommending defaults

Run one production-path Large LoRA adaptation on the pinned existing split, with
the original checkpoint hashes and decoding protocol. First match the experimental
unmerged/merged/exported predictions; then score within/across-bird F1, training
peak and warmed update time. A proposed regression gate is no more than 0.5
percentage-point F1 loss versus the recorded recipe on either test; this is an
engineering threshold, not a statistical confidence bound. Investigate export
rounding separately from training variance.

Benchmark the candidate Tiny/Base/Small/Large inference artifacts on the same
host/input, one warmup and five timed repeats in fresh processes. Keep each
host/backend separate. Record actual dtype, generated lengths, cap hits, event
F1, checkpoint/source hashes, GPU allocated/reserved or MPS live/driver memory,
and process RAM. Finish missing matched Small CUDA conditions only if needed for
the target release: the local 16 October receipt says its private input repository
still lacks required files, so those results are not complete.

Run relevant unit tests and CLI/GUI training/prediction integration tests. Keep
checkpoint-backed regressions under `tests/regression/`. Test an additional
independent recording/species and more seeds before turning Large LoRA into a
general default or selecting one universal backbone. Small and Tiny LoRA need
their own validation; Large success does not establish their best recipe.

## Review verification

This review recomputed all six Large LoRA/control held-out F1 scores and verified
their recorded result hashes/protocols; checked all nine paired Base pooled F1
scores and its three vocabulary-only equalities; and checked twelve Tiny/Small
vocabulary-only score pairs for exact F1 agreement. Existing measurements were
reviewed without rerunning fits or cloud jobs.

Focused tests could not collect in the requested `das-conformer` environment:
its command resolves to the base Python, which lacks torch, soundfile, DAS and
PEFT. No environment packages were changed for this documentation review. The
recorded earlier experiment test passes are historical evidence, not a fresh pass.
