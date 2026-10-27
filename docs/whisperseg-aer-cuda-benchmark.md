# WhisperSeg AER on Hugging Face Jobs

The launcher benchmarks all seven released checkpoints on `t4-small` (one T4,
16 GB GPU memory and approximately 15 GB system RAM). Hugging Face lists this
at $0.40/hour, with a one-hour timeout here. A full hour costs approximately
$0.40 in hardware charges; a shorter run costs less. A positive compute credit
balance is required. No 8 GB GPU is currently listed in the Jobs hardware menu.
Check [current pricing](https://huggingface.co/docs/hub/en/jobs-pricing).

## Run from this Mac

Create a token at <https://huggingface.co/settings/tokens> with repository read/write
and Jobs permissions. A Write token is the simplest choice. Log in locally;
do not paste the token into chat or put it in a script:

```bash
conda activate das-conformer
hf auth login
```

From the repository root:

```bash
# Prepare locally; no upload or compute charge.
python scripts/whisperseg_aer_hf_job.py

# Upload private inputs and launch one paid T4 job.
python scripts/whisperseg_aer_hf_job.py --submit
```

The default is the first 60 seconds of Chimichanga CH1, at its native 44.1 kHz.
Only this excerpt is uploaded. To use another recording or a longer excerpt:

```bash
python scripts/whisperseg_aer_hf_job.py --audio /path/to/recording.wav --seconds 300 --submit
```

The launcher packages the current local inference files, including uncommitted
changes, and records their hashes. It creates a new **private** dataset repository
in your personal account for inputs and results. The token is passed as an
encrypted job secret. Remote setup clones the CUDA PyTorch container's conda
environment into `das-conformer` and installs dependencies with `uv pip`.

The launcher prints the job and results links and saves them in
`runs/whisperseg-aer-cuda/job.json`. Monitor or stop the job with:

```bash
hf jobs inspect JOB_ID
hf jobs logs JOB_ID
hf jobs cancel JOB_ID
```

Download the results using the repository ID printed by the launcher:

```bash
hf download YOUR_NAME/whisperseg-aer-benchmark-RUN_ID --repo-type dataset \
  --include 'results/*' --local-dir runs/whisperseg-aer-cuda/downloaded
```

Results are saved after every model. A failed model is recorded and the remaining
models are attempted. Any model failure makes the final job exit with an error;
completed results remain available. A timeout can leave partial results.

## What is measured

- Fresh process per model; FP16 in evaluation/inference mode.
- One warmup on the entire excerpt, then five timed repetitions.
- Batch 1, four beams, three trials, maximum output length 128.
- Same spectrogram time step (2.5 ms) and minimum frequency (0 Hz) for every model.
- CUDA synchronization before and after timing. Downloads, model loading, WAV
  reading and warmup are excluded. Timings include feature extraction, transfers,
  generation and postprocessing.
- Median runtime, every repeat's runtime and predicted event count,
  audio seconds per wall-clock second (higher is faster), and real-time factor
  (wall-clock seconds per audio second; lower is faster).
- Peak PyTorch CUDA **allocated** and **reserved** memory across timed repeats,
  including resident model weights. Reserved memory includes the warmed allocator
  cache. These counters exclude CUDA context/driver allocations, so they can be
  lower than `nvidia-smi` memory use.
- Exact process RSS high-water mark from the OS, including imports, audio/model
  loading, warmup and inference. This is the worker's host RAM, not total machine
  memory or GPU memory. No previous model contributes to it.
- GPU name/capacity, Python/PyTorch/Transformers/CUDA versions, checkpoint revision,
  audio hash and local source hashes. All memory values are bytes.

`results.json` contains the complete metadata and measurements. `results.csv`
contains the comparison table. This measures speed and memory on one excerpt;
it does not measure segmentation accuracy or establish an 8 GB GPU requirement.

For an existing Linux CUDA installation with this repo's dependencies:

```bash
conda activate das-conformer
python scripts/benchmark_whisperseg_aer.py --audio /path/to/excerpt.wav --output-dir results
```

The direct benchmark uses the full supplied file. Its `--help` lists the inference
settings if a different protocol is needed.

## Remaining variants

`runs/whisperseg-aer-cuda-missing/models.json` selects seven variants absent from
the first run: original FP32 Base, Large animal VAD and Large multispecies;
trained compact FP16 Tiny/Small multispecies; and full-vocabulary FP32 Tiny/Small
architecture controls. The controls use pinned OpenAI speech weights, the original
Base WhisperSeg tokenizer/configuration and cropped encoder positions. They are
not trained segmenters, and their generated output lengths can differ substantially.
The previous seven AER variants are skipped. Original-versus-compact comparisons
include both vocabulary reduction and the change from FP32 to FP16.

Prepare the inputs with the selected local model directories beside the manifest:

```bash
python scripts/whisperseg_aer_hf_job.py \
  --models runs/whisperseg-aer-cuda-missing/models.json \
  --output-dir runs/whisperseg-aer-cuda-missing
```

The archive records every selected local checkpoint file's hash. The follow-up
job uses the identical audio bytes, inference source and settings from the first
run, with a two-hour timeout (at most about $0.80 in T4-small hardware charges).
Results retain each variant's dtype, vocabulary size, pinned source and whether
it is an architecture control.
