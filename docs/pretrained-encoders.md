# Frozen bioacoustic encoders

AVES2, Perch2, and BirdCODE can supply embeddings to the existing DAS decoders.
Select them in the GUI, Python API, or CLI. The raw frontend and encoder freezing
are selected automatically. Only the decoder is trained; DAS SSL pretraining is
not supported for these encoders.

| Encoder | Default model | Input rate | Context | Embedding size |
| --- | --- | --- | --- | --- |
| `aves2` | `esp_aves2_sl_beats_all` | 16 kHz | 10 s | 768 |
| `perch2` | [DAS Perch2 PyTorch](https://huggingface.co/DAS-DeepAudioSegmenter/perch2-torch) | 32 kHz | 5 s | 1536 |
| `birdcode` | [BirdCODE ConvNeXt backbone](https://huggingface.co/EarthSpeciesProject/sed-birdcode) | 32 kHz | 5 s | 1024 per frame |

Audio is averaged across channels, resampled to the encoder's rate, and split
into windows. For AVES2 and Perch2, each window produces one embedding; AVES2
temporal features are mean-pooled. `encoder_hop_seconds` controls output spacing
and defaults to the context length. A smaller hop gives overlapping contexts,
not sample-precise boundaries.

BirdCODE preserves 38 temporal frames per five-second window (7.6 Hz, about
132 ms per frame). It uses the released model's trained ConvNeXt features,
including its mel preprocessing and spatial upsampling, averaged over frequency.
These are the 1024-dimensional features **before** the 188,440-prototype species
head; DAS trains its own decoder on them. This integration does not run
BirdCODE's built-in 9422-species classifier or its detection postprocessing.
See the [paper](https://www.biorxiv.org/content/10.64898/2026.07.31.742086v1) and
[upstream implementation](https://github.com/earthspecies/sound-event-detection).

BirdCODE uses consecutive five-second windows, with `encoder_hop_seconds: 5`;
the output frame spacing remains `5/38` seconds. Keep that window hop fixed.
The final window is zero-padded and only frames covering valid audio are retained.

Windows start at each output frame's time and look forward for the context
duration. Labels and prediction timestamps use the output frame spacing. Windows at a
recording or DAS chunk boundary are zero-padded. Keep `num_time_steps` at least
one context window long, in **input recording samples**, and preferably several
windows long. This path suits broad vocal activity; use native DAS encoders for
brief pulses or precision beyond the chosen encoder's frame spacing.

## Cache embeddings, then train

In the GUI, choose an encoder and use the **Embedding model** dropdown:

- **Default model (Hugging Face)** uses the encoder's published default. AVES2's native loader requires AVEX on Python 3.11–3.13; on Python 3.14, use its local export below.
- **Hugging Face variant…** accepts a hosted `.pt2` embedding model, including a fine-tuned export. Paste its file link or `owner/repository/model.pt2`.
- **Local file…** opens a file picker for your own `.pt2` embedding model.

The same sources work through `encoder_checkpoint` in YAML, the API, and the CLI:

```yaml
encoder_checkpoint: hf://your-account/your-encoder@v1/model.pt2
# Or a local path: /path/to/model.pt2
```

Hugging Face files use the normal Hub download cache. Configs and trained DAS
checkpoints retain the hosted reference, so they can be moved to another machine.
For private repositories, log in with `hf auth login`. A revision after `@` is
optional; use a commit or tag to keep the chosen weights reproducible.
Embedding-cache keys include the downloaded file's contents.

Custom exports must include `das_embedding.json` with `sample_rate`,
`window_seconds`, and `embedding_dim`, as written by the export scripts below.
Match these settings to the encoder; BirdCODE exports must preserve its 38
temporal frames per window. A trained DAS decoder `.ckpt` belongs in **Initial
model** or **Checkpoint**; it is not an exported embedding backbone.

Install optional extraction dependencies with `uv pip install '.[embeddings]'`.
Perch2 uses `timm` and PyTorch Hub to load `janclemenslab/perch2_torch`, which
downloads your weights from Hugging Face. DAS calls its `embed()` method to skip
the classification head. No ONNX runtime is needed for Perch2. On Python 3.14,
AVES2 needs the export described below; its .pt2 inference needs only PyTorch.
BirdCODE uses the existing PyTorch, torchaudio, Transformers, and Hugging Face
dependencies; no additional package or conversion environment is required.

```sh
das embed --config configs/perch2-embeddings.yaml --data-dir /path/to/audio
das train --config configs/perch2-embeddings.yaml --data-dir /path/to/audio --output-dir /path/to/run
das predict --data-dir /path/to/audio --checkpoint /path/to/run/checkpoints/model.ckpt
```

Equivalent Python entry point: `das.embed(config)` followed by `das.train(config)`.
Use `configs/aves2-embeddings.yaml` for AVES2. For BirdCODE:

```sh
das embed --config configs/birdcode-embeddings.yaml --data-dir /path/to/audio
das train --config configs/birdcode-embeddings.yaml --data-dir /path/to/audio --output-dir /path/to/run
```

The cache stores one float32 `.npy` array per distinct audio window (38 × 1024
for BirdCODE, one embedding for the other encoders). Its key
includes the waveform, input rate, model settings, and the contents of a supplied
encoder checkpoint. Repeated runs reuse these files without loading the encoder.
Training still reads audio to identify windows; it does not repeat neural
inference for cache hits. New windows, including changed chunk/split boundaries,
are computed as needed. Unannotated audio is supported by `das embed`.

Use the same cache directory and settings for extraction and training. If using
AVEX's built-in model loader, Perch2, or BirdCODE, use a new cache directory after changing
the runtime or upstream weights; exported files are invalidated automatically by their hash.
Keep recording-level train/validation/test splits to avoid overlap between splits.

DAS decoder checkpoints store the encoder settings, not the external weights.
Hosted `.pt2` files are downloaded automatically; keep local exports available
when moving a decoder checkpoint.
AVES2, Perch2, and BirdCODE run on CPU or Apple MPS; all were checked on Python 3.14.
Their presets use `accelerator: auto`, which selects MPS on supported Macs.
Set `accelerator: mps` explicitly to require it. Exported graphs are moved with
PyTorch's `move_to_device_pass`, including device settings embedded in the graph.
CUDA can also be selected but was not tested here. Audio resampling and disk
caching remain on CPU.

## AVES2 on Python 3.14

The default is [ESP-AVES2 BEATs-all](https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-all),
pretrained on AudioSet and then trained on a mixture of bioacoustics and AudioSet.
It returns 768-dimensional embeddings for decoder training. AVEX is the upstream
loading library; `aves2` is the DAS encoder option.

AVEX 1.3.0 supports Python 3.11–3.13. For Python 3.14, export it once from a
Python 3.13 environment with AVEX and the same PyTorch version as DAS:

```sh
/path/to/python3.13 scripts/export_avex.py ~/.cache/das/avex/avex_aves2.pt2
```

The AVES2 preset points to that local export. In the GUI, load the preset or
select **aves2 → Embedding model → Local file…** and choose the `.pt2` file. On supported
Python versions, omit `encoder_checkpoint` to load AVES2 through AVEX directly. Other AVEX models can be
selected with `encoder_model_id`; match `encoder_sample_rate`,
`encoder_window_seconds`, and `encoder_embedding_dim` to that model/export.

The export was verified on Python 3.14.3 with PyTorch 2.13.0. CPU temporal features
match the saved upstream reference exactly; the maximum MPS difference was
approximately `3e-5` on that synthetic input. The export uses stride-safe reshapes
for MPS; regenerate older exports if they raise a tensor-view error. Its fixed
input shape is `(1, 160000)` and its temporal output is `(1, 496, 768)`, which DAS
mean-pools for each window. Run the reference checks with:

```sh
DAS_AVES2_EXPORT=~/.cache/das/avex/avex_aves2.pt2 pytest tests/regression/test_aves2_export.py
```

## Other models worth comparing

[BirdNET](https://birdnet-team.github.io/BirdNET-Analyzer/best-practices/embeddings.html)
is the next useful independent baseline. AVEX already offers BEATs, EAT,
EfficientNet, and BirdMAE families through its
[model library](https://github.com/earthspecies/avex/blob/main/docs/supported_models.md),
so compare those through the AVEX loader before adding separate integrations.
