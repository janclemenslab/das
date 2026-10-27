# Fly waveform Conformer experiment

## Result

A learned waveform filterbank can feed the existing Conformer and detects fly
pulses well. An animal2vec-style Sinc filterbank plus Conv1d stack produced the
best overall WER, although mel retained the best pulse and frame-level scores.
The remaining shared failure is sine-bout boundary decoding, not pulse
visibility alone.

The useful raw paths are `frontend_type: conv` and `frontend_type: sinc`: both
learn features directly from waveform samples before temporal downsampling. A
literal `frontend_type: raw` Conformer is not useful for 10 kHz audio because
attention would operate on every sample and scale quadratically, while the
encoder would receive only one feature per time step.

## Data inspected

The folder contains three mono, 10 kHz WAV files:

| recording | duration | pulses | sine bouts | annotation marker |
|---|---:|---:|---:|---:|
| `20170316125103_chn4.wav` | 2,338 s | 1,153 | 71 | 310 s |
| `20170317142609_chn15.wav` | 2,471 s | 1,316 | 13 | 530 s |
| `20170323133129_chn8.wav` | 2,485 s | 10,100 | 821 | 360 s |

The loader correctly treats pulse as an event and sine as a segment. Pulse
annotations are points; their median inter-pulse interval is about 37--42 ms.
Sine bouts have median durations of 165--488 ms and maxima near 1.9 s.

Important caveat: the channel-8 CSV has an `annotation_end` marker at 360 s but
contains 9,616 pulse/sine rows after that marker. Current training intentionally
ignores those later rows. Move/remove the marker only if the later annotations
are known to be complete; this experiment did not silently override it.

## Dense-decoder comparison

All models used the same 64-dimensional, two-layer Conformer and 1,000 training
steps. The cross-recording split trained on channel 8, validated on channel 4,
and tested on the unseen channel 15 recording.

| frontend | resolution | parameters | pulse frame F1 | sine frame F1 | pulse event F1 | sine interval F1 (+/-10 ms) | WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned waveform conv (64-sample kernel) | 1 ms | 101K | 0.83 | 0.71 | 0.93 | 0.02 | 0.202 |
| Sinc filterbank + Conv1d stack (79-sample kernel) | 1 ms | 191K | 0.79 | 0.83 | 0.91 | 0.03 | **0.178** |
| SongExplorer-style raw ConvResNet | 1 ms | 192K | 0.75 | 0.87 | 0.92 | 0.03 | **0.100** |
| mel (256-sample FFT) | 4 ms | 171K | 0.88 | 0.88 | 0.95 | 0.02 | 0.237 |

On the easier within-recording split, pulse event F1 was 0.94 for waveform and
0.96 for mel. The Sinc frontend improves sine frame F1 and overall sequence
error over the simpler learned waveform convolution, but does not improve pulse
localization. All three dense models fragment long sine bouts: increasing gap
filling did not improve sine interval F1.

The SongExplorer-style frontend was trained on `pan.local` for the same 1,000
updates. It applies a 9-sample learned waveform convolution and three residual
two-convolution blocks with dilations 1, 2, and 4 before a learned stride-10
downsampler. This gives each 1 ms feature about 13.9 ms of raw-waveform context.
It substantially reduces sequence errors and keeps pulse event F1 competitive,
but trades away pulse frame accuracy and does not fix sine-bout fragmentation.
The result supports retaining a high-resolution local waveform branch before
the Conformer rather than replacing the Conformer with SongExplorer's
window-classification workflow.

## Pulse/sine specialist split

The different time scales justify separate prediction branches. On the same
cross-recording split, a pulse-only model using the SongExplorer-style
ConvResNet frontend at 1 ms followed by a compact TCN reached pulse frame F1
0.86 and pulse event F1 0.94 (precision 0.94, recall 0.95; WER 0.011). This is
better than the joint ConvResNet-Conformer model's 0.75 frame F1 and 0.92 event
F1.

A matched classical DAS baseline was also trained for 1,044 updates: raw
waveform input, 32 filters, three TCN stacks, kernel size 16, dilations 1--16,
and no dropout or post-optimization. It reached pulse frame F1 0.78 and event
F1 0.78 (precision 0.67, recall 0.94; WER 0.393). The old TCN detects most
pulses but produces too many events. The learned high-resolution frontend is
therefore useful even when the pulse context encoder is convolutional.

### Sine re-evaluation at +/-30 ms

A sine interval counts as correct when both its onset and offset are within
30 ms of the same reference interval. The table below is a micro-average over
the two held-out recordings (channels 4 and 15). Checkpoint outputs and
postprocessing were left unchanged, so the collar is the only scoring change.

| model | F1 at +/-10 ms | F1 at +/-30 ms | matches / predictions / references at +/-30 ms |
|---|---:|---:|---:|
| joint ConvResNet--Conformer, early | 0.060 | 0.184 | 40 / 351 / 84 |
| joint ConvResNet--Conformer, 1,000 steps | 0.038 | 0.106 | 28 / 446 / 84 |
| joint timestamp decoder | 0.000 | 0.000 | 0 / 0 / 84 |
| sine mel--Conformer, early | 0.048 | 0.255 | 21 / 81 / 84 |
| sine mel--Conformer, tuned | 0.049 | 0.256 | 21 / 80 / 84 |
| sine classical DAS TCN | 0.070 | 0.185 | 29 / 229 / 84 |
| sine timestamp decoder | 0.000 | 0.000 | 0 / 43 / 84 |
| sine mel--Conformer, post-marker channel-8 annotations | 0.037 | 0.199 | 16 / 77 / 84 |
| sine ConvResNet--Conformer, 4 s context | **0.170** | **0.424** | **35 / 81 / 84** |

The relaxed collar changes the magnitude but not the ranking. The long-context
raw-waveform model remains the best saved checkpoint. Its held-out precision is
0.432 and recall is 0.417. The two timestamp models are not merely outside the
tighter collar: one emits no sine intervals and the other emits 43 intervals
without a match even at 30 ms.

The first sine-only model used a 4 ms mel frontend and a two-layer Conformer.
It reached sine frame F1 0.78. At the adopted +/-30 ms collar its tuned version
has interval F1 0.284 on channel 4 and 0.087 on channel 15, or 0.256 across both
held-out recordings. The original postprocessing tuner reported WER near zero
because a one-class label sequence does not measure boundary placement. Tuning
now ranks candidates by time-matched interval F1, with WER only as a tie-breaker;
this selected 80 ms gap filling and 92 ms minimum duration. Sine still benefits
from more temporal resolution as well as long context.

A sine-only ConvResNet--Conformer now provides both: the waveform frontend has
a 1 ms hop, while four-second chunks let the Conformer see complete bouts. It
retained sine frame F1 0.79 and substantially improved strict interval timing.
With 80 ms gap filling and a 92 ms minimum duration it matched 8 of 71 intervals
on channel 4 and 6 of 13 on channel 15 at the old +/-10 ms collar. At +/-30 ms,
it matches 26 of 71 intervals on channel 4 (68 predictions, interval F1 0.374)
and 9 of 13 on channel 15 (13 predictions, interval F1 0.692). Across the two
held-out recordings that is 35 matches, 81 predictions, 84 references, and
interval F1 0.424. The exact architecture and training settings are in
`configs/stern2017-sine-convresnet-long.yaml`; the reported fit was warm-started
from the earlier joint ConvResNet--Conformer checkpoint.

Several plausible boundary refinements did not generalize. Standalone onset and
offset event detectors collapsed under the extreme background imbalance;
positive weighting and boundary-centered sampling produced weak false peaks.
A band-passed waveform-envelope refiner reduced held-out accuracy. At +/-30 ms,
a jointly fine-tuned activity-plus-boundary head reaches interval F1 0.343 on
channel 4 and 0.560 on channel 15 (0.376 aggregate). This improves that model's
own dense output (0.339 aggregate), but remains below the raw long-context
model. A frozen boundary head on the 1 ms model reduces aggregate F1 from 0.424
to 0.412. Postprocessing selected on the training recording behaved similarly.
The simple 1 ms dense output is therefore the current sine baseline; boundary
heads should only be revisited with more independent recordings or species.

Removing channel 8's 360-second `annotation_end` marker exposed 821 rather than
72 existing sine bouts, but training on those post-marker rows reduced held-out
interval F1 to 0.199 at +/-30 ms. Those rows should not be treated as equivalent
additional supervision without checking how they were produced.

## All-file Stern2017 specialist fits

The complete `dat.raw` corpus contains 77 recordings: 66 with pulse and sine
annotations and 11 fully annotated negative recordings. With seed 0, whole
files were split 54/12/11 into train/validation/test. The partitions contain
474,484/102,473/104,447 pulse events and 35,308/6,554/8,836 sine bouts,
respectively. Seven/two/two negative recordings fall in the three partitions.
The three recording sessions contribute channels to multiple partitions, so
this is a file-held-out estimate rather than a recording-session-held-out one.

The all-data sine model uses the 1 ms ConvResNet frontend, two-layer Conformer,
four-second input, and dense sine/no-sine head. Its validation-selected 20 ms
gap fill and 80 ms minimum bout give interval F1 0.4666 on validation and 0.4417
on test at the +/-30 ms boundary collar (4,185 matches, 10,115 predictions, and
8,836 references). Test frame F1 is 0.81. A further 1,000 training updates
reduced validation interval F1 to 0.4306 at identical settings and were rejected.

A two-threshold activity gate improves the retained checkpoint without another
fit. Candidate regions use a low threshold to preserve boundaries, but are kept
only if they contain a high-confidence frame. Validation selected low/high
thresholds 0.2/0.9, 20 ms gap fill, and a 100 ms minimum bout. Validation
interval F1 rises to 0.5541; the frozen settings reach 0.5225 on test (4,760
matches, 9,384 predictions, and 8,836 references). One negative test channel
still produces 1,120 false bouts while the other produces none, indicating a
remaining channel-specific false-activity problem.

The all-data pulse specialist uses the ConvResNet frontend and compact classical
DAS TCN. At the +/-10 ms event collar it reaches validation F1 0.8731 (86,338
matches, 95,294 predictions, and 102,473 references) and test F1 0.8180 (85,224
matches, 103,916 predictions, and 104,447 references). Positive test-file F1
ranges from 0.812 to 0.910. As for sine, one negative test channel is pathological:
it produces 10,519 false events while the other produces none. The configs are
`configs/stern2017-all-sine.yaml` and `configs/stern2017-all-pulse.yaml`.

The Sinc frontend adapts the official
[animal2vec feature extractor](https://github.com/livingingroups/animal2vec/blob/main/configs/MeerKAT/a2v_large_pretrain_best.yaml)
to this smaller model. At 10 kHz it uses 127 learnable, mel-initialized Sinc
band-pass filters with a 79-sample kernel, then 64-channel Conv1d blocks with
strides 5 and 2. Layer normalization and GELU follow each layer, producing 1 ms
features for the unchanged Conformer. Animal2vec uses the same Sinc-first,
progressively strided pattern at much larger width and a 5 ms final resolution.

## Mixed timestamp decoder

The native timestamp decoder is the appropriate small analogue of a Whisper
decoder for this task. Reusing the full pretrained Whisper decoder would add a
large speech-language model whose vocabulary and timestamp priors do not match
two fly-song labels or millisecond events.

Fly data exposed and motivated three fixes:

1. Overlapping within-file split intervals duplicated timestamp targets. The
   intervals are now merged before annotations are clipped; real nested and
   overlapping annotations remain intact.
2. The tied token/output embedding was initialized with unit variance, causing
   an initial loss of 27.7 instead of roughly `log(vocabulary)`. Scaled
   initialization plus a final decoder layer normalization reduced it to 6.69.
3. Teacher forcing processed the full configured token capacity even for quiet
   chunks. It now trims each batch to its longest real target. Four-second fly
   chunks require capacity for at most 66 annotations (`max_length: 200`).

After 500 steps, held-out loss fell from 6.69 to 4.55 and exact token accuracy
reached 38%, but greedy decoding still emitted immediate EOS on a positive
chunk. The decoder is therefore not ready for use.

## Activity-gated timestamp decoder

The timestamp decoder now has a one-logit-per-frame activity head trained from
the existing event and interval triples. At generation time, its maximum valid
frame gates first-token EOS, and its frame logits bias onset timestamp logits.
This adds only 65 parameters to the 64-dimensional model.

After 500 updates, the best held-out total loss was 4.96 (token plus activity),
activity loss was 0.23, and token accuracy was 37%. At the default zero-logit
threshold, frame activity precision/recall/F1 was 0.92/0.29/0.44. The gate
detected 35 of 74 positive validation chunks and correctly rejected all nine
empty chunks. Most importantly, the positive example that previously emitted
immediate EOS now emitted a pulse timestamp; three inspected empty chunks
emitted none.

This fixes the immediate-EOS failure but not timestamp decoding as a whole. On
the example with nine pulses and one sine interval, the model emitted one pulse
at frame 105 rather than the complete sequence. Experiments with balanced
per-batch activity loss and separate/max-pooled chunk gates increased recall but
made every or several empty chunks active and damaged frame localization, so
the smaller dense-head version was retained. The next decoder experiment should
calibrate the activity threshold on validation data and prevent duplicate or
non-increasing event timestamps before increasing model size.

## Toward a raw-waveform fly foundation model

This is feasible as a compact transferable fly encoder, although the available
data do not yet justify a Whisper-scale model. Not all existing foundation
models use spectrograms: wav2vec 2.0, HuBERT, AVES, and animal2vec take raw
waveforms. Their usual 20 ms speech/audio stride is too coarse for fly pulse
centres; [animal2vec](https://arxiv.org/abs/2406.01253) is the closest design
precedent because it combines a SincNet-like waveform frontend, 5 ms features,
and teacher-student latent prediction for rare animal calls.

A useful first model would have:

- a learned Conv1d or Sinc frontend retaining 1 ms local pulse features;
- a 4--5 ms, 10--30M-parameter Conformer context encoder;
- EMA teacher-student masked latent prediction with 2--100 ms mask spans;
- downstream pulse-centre, dense activity, sine-boundary, and optional timestamp
  heads; and
- gain, polarity, shift, mild filtering, and background-mixing augmentation,
  but no strong time stretching because pulse shape and IPI are biological
  variables.

The current repository already supplies recursive multi-recording loading, a
waveform Conv frontend, a Conformer, masked pretraining scaffolding, and
frontend/encoder checkpoint transfer. Its current SSL loss reconstructs detached
features from the same trainable frontend, however, so it has a zero-feature
collapse solution. Treat it as a baseline only; check feature variance/effective
rank, then replace it with an EMA target encoder before scaling.

The strongest open seed corpus is the
[SongExplorer 21-species collection](https://figshare.com/articles/dataset/Audio_recordings_of_21_species_of_Drosophila_10_of_which_are_annotated/14328425):
10 species have pulse/sine annotations and all 21 provide raw audio. It can be
combined with the [large multi-condition D. melanogaster
collection](https://doi.org/10.25378/janelia.24707544.v4), the
[Stern recordings](https://datadryad.org/dataset/doi:10.5061/dryad.td223), and
the [manual Stern/Kyriacou pulse
reannotations](https://datadryad.org/dataset/doi:10.5061/dryad.80c1f). This is
more than 50 GB of available waveform material, but annotation completeness and
label ontologies differ.

The decisive evaluation is leave-one-species-and-recording-rig-out transfer:
compare training from scratch, a frozen linear probe, and full fine-tuning with
1, 5, and 10 annotated minutes. Report pulse event F1 at +/-2 ms, onset error,
sine interval F1/IoU, PR-AUC, and false positives per hour. Scale the model only
if pretraining improves unseen-species or low-label performance over matched raw
and mel baselines.
