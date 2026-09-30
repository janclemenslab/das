# DAS
DAS 1.0a3 segments and annotates audio with Conformer, TCN, TweetyNet, and WhisperSeg models.


## Installation

```shell
conda create -y -n das python=3.12 uv
conda activate das
uv pip install "das[gui]==1.0a3"
```

For development from a source checkout, install the test and docs extras too:

```shell
uv pip install -e ".[dev,gui,doc]"
```

The GUI uses xarray-behave. Run `das` or `das gui` to open the single-file annotation view and its Train and Predict dialogs.

## Usage
Python API:

```python
import das

checkpoint = das.train(
    data_dir="/path/to/audio",
    output_dir="/path/to/run",
)

written_files = das.predict(
    data_dir="/path/to/audio",
    checkpoint=checkpoint,
    existing_annotations="merge",
)

annotations = das.predict(
    audio=waveform,
    samplerate=16_000,
    checkpoint=checkpoint,
)
```

Prediction writes each annotation CSV next to its audio file by default. Set `output_dir="/path/to/predictions"` to collect them in one directory. Set `output_dir=""`, or pass raw `audio`, to return annotation DataFrames instead of writing prediction CSV files. Training still writes to `./` by default.
Prediction handles existing output CSV files with `existing_annotations="skip"`, `"overwrite"` (default), or `"merge"`. The CLI equivalent is `--existing-annotations`.

Training reads annotated WAV folders directly; no dataset generation step is needed. It also accepts existing DAS `.npy`, H5, and Zarr training datasets. Other audio formats supported by the current DAS release remain available through the existing dataset-generation path.

Minimal CLI:

```shell
das train \
  --data-dir /path/to/audio \
  --output-dir /path/to/run \
  --config /path/to/train.yaml \
  --frontend mel \
  --batch-size 16
```

```shell
das predict \
  --data-dir /path/to/audio \
  --checkpoint /path/to/run/checkpoints/best-epoch=000.ckpt \
  --output-dir /path/to/predictions \
  --config /path/to/predict.yaml \
  --evaluate \
  --syllable-tolerance-ms 20
```

```shell
das predict \
  --data-dir /path/to/audio \
  --checkpoint /path/to/run/checkpoints/best-epoch=000.ckpt \
  --output-dir /path/to/predictions \
  --config /path/to/predict.yaml \
  --existing-annotations merge \
  --output-suffix _frames.csv
```

CLI flags and YAML use the same flat field names. Config files can be partial, and `--config` can be provided multiple times. The merge order is defaults, then config files in the order provided, then explicit CLI flags.

`--config` also accepts built-in config names: `fly-pulse`, `fly`, `zebra-finch`, and `tweetynet`. `fly-pulse` selects the 1 ms raw-waveform ConvResNet plus compact TCN. They load through the same merge path as YAML files:

```shell
das train \
  --config fly-pulse \
  --data-dir /path/to/audio \
  --output-dir /path/to/run
```

Legacy DAS models saved as `*_model.h5` plus `*_params.yaml` can be used with the same `predict` command by pointing `--checkpoint` at the shared trunk path:

```shell
das predict \
  --data-dir /path/to/audio \
  --checkpoint /path/to/test \
  --output-dir /path/to/predictions
```

They can also be converted once to a native Torch checkpoint that loads through `DASModel.load_from_checkpoint()`:

```shell
das convert-legacy \
  --checkpoint /path/to/test \
  --output /path/to/test.ckpt
```

The converted checkpoint can then be passed to `das predict` like any native checkpoint.

WhisperSeg uses DAS `.ckpt` checkpoints for training and prediction. Training requires `--initial-model /path/to/converted-whisperseg.ckpt`; original WhisperSeg `.pt` files and Hugging Face model IDs are not loaded directly.

The self-contained `whisperseg-aer/v1` `.pt` bundles can be repacked once, without retraining:

```shell
python -m das.whisperseg.convert /path/to/whisperseg-aer.pt /path/to/whisperseg-aer.ckpt
```

For native checkpoints, `das predict` uses the saved chunk length, stride, and class names from the trained model. Legacy DAS models use the saved legacy window size, stride, and class labels from the model params. Evaluation works through `das predict --evaluate`, with optional `--split train|val|test` for split-backed datasets.

Print a starter config for a subcommand:

```shell
das train --print-config
```

Save the effective config for a run:

```shell
das train \
  --data-dir /path/to/audio \
  --output-dir /path/to/run \
  --encoder tcn \
  --save-config /path/to/run-config.yaml
```

Segment detection uses adjustable low and high hysteresis thresholds. The CLI, YAML config, and GUI Train and Predict dialogs expose these thresholds, event thresholds and spacing, and manual postprocessing settings.

Built-in TCN encoder example:

```shell
das train \
  --data-dir /path/to/audio \
  --output-dir /path/to/run \
  --encoder tcn \
  --encoder-hidden-size 32 \
  --encoder-num-layers 4 \
  --encoder-dilations [1,2,4,8,16]
```

Example flat config:

```yaml
mode: train
data_dir: /path/to/audio
output_dir: /path/to/run
batch_size: 16
num_time_steps: 2048
frontend_type: raw
encoder_type: tcn
encoder_hidden_size: 16
encoder_num_layers: 2
encoder_dilations: [1, 2, 4, 8]
decoder_type: linear
learning_rate: 0.01
num_epochs: 20
```
