# Install and train

Install the DAS 1.0a2 pre-release with its GUI:

```shell
conda create -n das python=3.12 uv
conda activate das
uv pip install "das[gui]==1.0a2"
```

For annotated WAV recordings, put each audio file and its annotation CSV in one folder. The default annotation name is `<audio-stem>_annotations.csv`. A CSV has `name`, `start_seconds`, and `stop_seconds` columns. Train directly from that folder:

```shell
das train --data-dir /path/to/wavs --output-dir /path/to/run --config fly
```

Available built-in YAML presets are `fly`, `fly-pulse`, `zebra-finch`, and `tweetynet`. A YAML file can override any CLI config field. Inspect a starting configuration with `das train --print-config`.

Predict and optionally compare the result with annotation CSV files:

```shell
das predict --data-dir /path/to/wavs --checkpoint /path/to/run/checkpoints/model.ckpt --evaluate
```

Set `--segment-threshold-low` and `--segment-threshold-high` to control hysteresis for segments. Use `--event-threshold` for events. `--fill-gap-ms`, `--min-syllable-ms`, `--event-dist-min-ms`, `--event-dist-max-ms`, and `--syllable-postprocessor` provide manual postprocessing. These settings are also in the GUI dialogs.
