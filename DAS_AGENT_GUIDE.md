# Using DAS 1.0 with an AI agent

Use this guide for the DAS 1.0 CLI and Python API. Confirm the installed version with `das version` and inspect current options with `das train --help` or `das predict --help`. The examples below target the 1.0a1 interface in this repository.

## Before running anything

1. Ask for the task (train, predict, or evaluate), the audio location, and whether annotations already exist. For WhisperSeg training, also ask for a **DAS WhisperSeg `.ckpt`** initial model.
2. Inspect a few audio files for sample rate, duration, and channels. Inspect the annotation names, segment durations, and a spectrogram before choosing frequency or time settings. Do not infer the call band from the species name alone.
3. Keep training, validation, and test recordings separate when possible. Do not put predictions into the ground-truth folder unless the user wants that. Review a few predicted clips and errors before processing the full collection.

Training from an audio folder expects each annotated recording to have a matching CSV, such as `bird.wav` and `bird_annotations.csv`:

```csv
name,start_seconds,stop_seconds
call,0.42,0.57
call,1.13,1.29
```

Times are seconds. A point event has equal start and stop times. If only part of a recording was annotated, bound that region with `annotation_start` and `annotation_end` rows; otherwise DAS treats the whole file as annotated. Keep a separate held-out recording for a meaningful check.

## Choose a model

| Data or goal | Start with | Reason |
| --- | --- | --- |
| General labeled segments, no pretrained checkpoint | Default mel + Conformer | Simple general-purpose DAS 1.0 baseline. |
| Brief pulses or point events, especially fly song | `fly-pulse` (or `fly` for both pulse and sine) | Native waveform/TCN models provide fine timing; `fly-pulse` selects only the `pulse` label. |
| Zebra finch syllables | `zebra-finch`; compare `tweetynet` if useful | Existing task presets avoid guessing architecture settings. |
| Animal vocal activity with an existing DAS WhisperSeg checkpoint | WhisperSeg | Fine-tune a pretrained segmenter; it needs more compute and a `.ckpt` initial model. |

Use validation results and inspected predictions to choose between candidates. WhisperSeg is a segment model; for zero-duration events, start with a native DAS model. AVES2, Perch2, BirdCODE, `das embed`, and the timestamp decoder are **not included in 1.0a1**.

## Train, predict, evaluate

Print the effective config before a run; CLI flags override YAML fields, and later `--config` files override earlier ones:

```sh
das train --config zebra-finch --print-config
das train --config zebra-finch --data-dir /path/to/annotated-audio --output-dir /path/to/run --save-config /path/to/run-config.yaml
```

Training returns/writes a checkpoint under the run directory's `checkpoints/`. Use the actual `.ckpt` path from that run:

```sh
das predict --data-dir /path/to/new-audio --checkpoint /path/to/run/checkpoints/ACTUAL.ckpt --output-dir /path/to/predictions --existing-annotations skip
das predict --data-dir /path/to/held-out-annotated-audio --checkpoint /path/to/run/checkpoints/ACTUAL.ckpt --output-dir /path/to/evaluation-predictions --evaluate
```

`--evaluate` compares predictions to available annotations. By default, prediction writes `_annotations.csv` beside each audio file and overwrites an existing output. Use `--output-dir` and `--existing-annotations skip` or `merge` when appropriate. For scripts, `das.train(data_dir=..., output_dir=...)` returns the checkpoint path; `das.predict(data_dir=..., checkpoint=..., output_dir=...)` returns written paths. With raw `audio=waveform`, also pass `samplerate=...`; prediction returns a DataFrame.

## WhisperSeg: frequency and temporal settings (`freqs` and `temp`)

The CLI names are `--min-frequency`, `--frequency-scale`, and `--spec-time-step`; the YAML/Python names use underscores. These are **WhisperSeg** settings, separate from the native `frontend_fmin`, `frontend_fmax`, and `frontend_hop_seconds` fields.

| Setting | Starting recommendation | Change it when |
| --- | --- | --- |
| `min_frequency` (Hz) | Leave unset. Training then uses 0 Hz; prediction uses the checkpoint's saved value. | Low-frequency noise dominates and the lowest target call frequency is known. Set it **below** the lowest useful call component, with margin. |
| `frequency_scale` | Keep `1.0`. | The useful call band is badly placed for the model's input representation. Values above 1 shift audio frequency upward; values below 1 shift it downward. Compare against 1.0 on held-out clips and use the same value for training and prediction. This is not ordinary audio resampling. |
| `spec_time_step` (seconds per spectrogram column) | Leave unset. Training chooses a value from median labeled segment length; prediction uses the checkpoint's saved value. | The default window is too short/long for the calls or boundary precision is poor. With the default 1,000 columns, `0.001` spans about 1 s, `0.0025` about 2.5 s, and `0.005` about 5 s. Smaller steps give finer timing but less context. |

For a **new** WhisperSeg run, set frequency and time controls on training, then keep them consistent for prediction. First try the automatic settings. Change one parameter at a time based on a spectrogram and held-out evaluation. In particular, if you trained with `frequency_scale` other than 1, pass that same value at prediction; the CLI default is 1.0.

```sh
das train --encoder whisperseg --initial-model /path/to/initial-whisperseg.ckpt --data-dir /path/to/annotated-audio --output-dir /path/to/whisperseg-run
das predict --checkpoint /path/to/whisperseg-run/checkpoints/ACTUAL.ckpt --data-dir /path/to/new-audio --output-dir /path/to/predictions
```

`--initial-model` must point to an existing local DAS WhisperSeg `.ckpt`; use a DAS WhisperSeg `.ckpt` for `--checkpoint` too. Raw Hugging Face model IDs and WhisperSeg `.pt` bundles are unsupported.

For the precise flags and dataset rules, see [the README](README.md), [the 1.0 quickstart](docs/alpha/quickstart.md), and [WhisperSeg's parameter explanation](https://github.com/nianlonggu/WhisperSeg/blob/master/docs/DatasetProcessing.md).
