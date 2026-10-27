# WhisperSeg large distillation — complete

Started 2026-09-16. The user requested large-model distillation with several Conformer sizes and a fixed decoder, followed by comparison with WhisperSeg base, WhisperSeg large, and the existing base-distilled hybrid. Use the original base-distillation dataset and also test nightingale. The user explicitly authorized pan.local and, on September 18, the separate pan.uol host at `janc@100.120.244.47`.

## Live job

- Remote root: `/Users/janc/segmentation-refits/experiments/whisperseg-large-distillation-20260916`.
- Local artifacts: `runs/whisperseg-large-distillation-20260916/`.
- The primary queue controller **65464** and sleep guard **65465** completed successfully after finishing all twelve evaluations. Smallest trainer **41549** and all later evaluation processes exited normally. Current pan.local `status.json` is `complete`; no training or evaluation remains active.
- Runner: `scripts/whisperseg_large_distillation.py`, copied to the remote root. It waits for the existing smallest student, waits for verified transfers of both middle and largest students from pan.uol, then runs all twelve evaluations on pan.local. After the smallest fit ends, the waiting controller uses no GPU, leaving pan.local free for other work until evaluation. A successful stage is skipped on a subsequent invocation. Failed stages stop the queue; inspect their logs before any recovery.
- Runtime: the existing `das-conformer` conda environment, PyTorch 2.11.0, Transformers 4.57.1. MPS with CPU fallback disabled. No environment packages were changed. An unrelated BirdCODE workload shares this host; preserve it.
- The waiting queue retains its `caffeinate -d -i -s -w PID` guard. The old queue controller 41548, conda wrapper 41532 and old guard 41533 have exited; this was an intentional queue handoff, not a failed or restarted training run.
- Final result, September 24: **3/3 student fits and 12/12 evaluations complete**. Smallest completed 50 epochs (best validation loss **0.7663395401**, epoch 50; **5,733,120 parameters**); see `large-distilled-562k/collection-verified.json`. Largest completed 50 epochs (best validation loss **0.7503854606**, epoch 50; **9,017,728 parameters**); see `large-distilled-4.01m/transfer-verified.json`. Middle completed 49 epochs by early stopping (best validation loss **0.7548543318**, epoch 44; **7,094,528 parameters**); its completion and transfer are recorded in `large-distilled-1.92m/transfer-verified.json`. The 12 evaluations have expected dataset counts and finite metrics; see `runs/whisperseg-large-distillation-20260916/comparison.md`.

Do not interrupt or restart a healthy run. Inspect the active log and process commands, not only the existence of a checkpoint. The first epoch takes roughly an hour plus validation based on a short shared-GPU timing check; the full sweep may take several days. Treat this as a rough estimate, not a measured whole-epoch runtime.

## Parallel training on pan.uol, 2026-09-18

- SSH: `janc@100.120.244.47`; hostname `pan`, Mac14,13, 32 GB RAM. This is a different machine from pan.local (Mac13,1).
- Run root: `/Users/janc/whisperseg-large-distillation-20260916`, outside Dropbox. Local copies of its job/status/logs are under `runs/whisperseg-large-distillation-20260916/pan-uol/`; never overwrite pan.local's root status with them.
- Python: `/Users/janc/miniconda3/envs/das-conformer/bin/python`; MPS, CPU fallback disabled, PyTorch **2.12.0**, Transformers 4.57.1. Both middle and largest students train here. pan.local trains only the smallest with `/Users/janc/miniforge3/envs/das-conformer/bin/python`, PyTorch **2.11.0**. No packages changed. **Include this training hardware/runtime difference in the comparison limitations. All twelve final evaluations remain on pan.local, with shared-hardware timing caveats.**
- Largest student `large-distilled-4.01m` began September 18 at 19:48:30 UTC. Wrapper **87936**, trainer **87938**, sleep guard **87937**. First 100/4,188 training batches passed with finite mean loss **6.0604**. Inspect that host's `job.json`, `status.json`, `runner.log`, model `train.log`, and `training_history.csv` (created after epoch 1).
- Preserve unrelated segmentation process 22180 and guard 22181. These PIDs are historical observations; verify commands before acting on any PID.
- `smoke-pan-uol.json` records matching 3,121 training annotation hashes/audio sizes, teacher SHA-256, initial decoder SHA-256, tokenizer IDs, exact split boundaries, 9,017,728 total parameters, and three finite MPS optimizer steps. The framewise initializer SHA-256 matches pan.local: `ce5af5f6a2e2dd83300e7b2074b90d5cc1aaf05f7c738ec6383b128b8b272065`. Frozen source hashes also match. Actual training parameters are unchanged.
- Only the queue controller on pan.local was replaced. Its original job/status and runner are preserved as `job-before-parallel.json`, `status-before-parallel.json`, and `runner-sequential.py`. Trainer 41549, its optimizer/RNG state, checkpoints, and frozen `source/` were untouched. The queue's `training-owner.json` files prevent duplicate smallest/largest runs. One queue ownership regression check and three integration checks passed.
- **Do not start the middle or largest student on pan.local.** Both have `training-owner.json` assigning them to pan.uol. Each queue stage waits for its own `transfer-verified.json`, which the collector must write only after successful training, complete artifact transfer, and SHA-256 checks. A `training_result.json` alone does not release either stage.
- On each pan.uol fit's completion, collect its `large-distilled-4.01m/` or `large-distilled-1.92m/` directory into the main local run and into pan.local's corresponding model directory, excluding mixed-rate data symlinks and temporary caches. Verify both final checkpoint hashes across pan.uol/local/pan.local and the tokenizer/config files; record them in `transfer-verified.json` on pan.local and locally. Preserve the primary `training-owner.json`. The primary controller proceeds automatically after both transfers are verified. Count completed models by unique model name, avoiding double-counting transferred copies.
- If either host's wrapper stops, inspect evidence and current processes before recovery. Do not rerun `handoff-queue.py` or launch `run-largest.py` over an existing partial fit. A dead externally owned trainer must be recovered explicitly; the primary queue intentionally waits instead of silently restarting it.

## Middle model moved to pan.uol, 2026-09-19

The user requested moving the middle distillation to pan.uol to free pan.local for other jobs. It had not started. On pan.local, the existing queue now reads an external ownership marker for this stage; no queue or trainer was restarted. Its previous metadata is preserved in `job-before-middle-move.json`.

- The middle model is queued behind the healthy largest model on pan.uol. Queue PID **56734**, sleep guard **56735**. It starts only after largest wrapper 87936 exits successfully and its completion result exists. If that dependency fails, the middle queue records failure and stops.
- Inspect **both** pan.uol root `job.json`/`status.json` (largest) **and** `large-distilled-1.92m/job.json`, `status.json`, `queue.log`, then `train.log` and `training_history.csv` once training starts. Root status becoming complete means only the largest fit completed; the middle fit has separate status. Missing largest PIDs/guard after successful completion is expected.
- `scripts/whisperseg_train_middle.py` is deployed as `run-middle.py` in the remote run root. The queued middle process uses no GPU. Its exact original five-layer training command was verified unchanged except paths, using the same already-verified initializer, decoder, teacher, source and dataset as the largest model.
- Two queue tests plus three integration checks passed. They check successful dependency ordering, refusal to start after failed largest training, and refusal to restart an attempted middle fit.
- The primary queue's `waiting_external` status after the smallest completes is expected and has no `child_pid`; inspect the pan.uol model statuses for progress. Its sleep guard remains active while it waits. Keep unrelated work untouched.

## Largest completed and middle started, 2026-09-21

Largest completed successfully at **10:56:54 UTC**; its wrapper 87936, trainer 87938 and guard 87937 exited normally. All 50 training/validation loss records are finite. Both final checkpoints contain 366 finite tensors; their configs match, and the FP16 tensors equal the FP32 checkpoint converted to FP16. The decoder remains one layer, width 512, FFN 2048; the Conformer has 10 layers. Tokenizer files match the smallest student.

All 14 completed files were copied to the main local run and pan.local. SHA-256 matched on all three hosts, including `best-model.ckpt` (`0b5d54f277c3586326ed6bdba547f737ceb9dfe644fba836ed75abcbd08a3a64`) and `das-whisper-hybrid-fp16.ckpt` (`82c33086d44c3213d57b65dc0d3ac3a9108257ef9cd26f03117e1f0b604b63b7`). Primary `training-owner.json` was preserved. Transfer markers were written only after verification. Do not recopy or retrain the largest model.

Middle started automatically at **10:56:57 UTC** with trainer **36958**, wrapper **56734**, guard **56735**. The original five-layer MPS training command is unchanged. Monitor its model-level status/history/logs; the pan.uol root status now correctly remains `complete` for the largest model. At 11:11 UTC the primary controller 65464 and guard 65465 remain alive, `waiting_external` for the middle model with no child process. Once the middle fit completes, transfer and verify it in the same way; its marker will release the existing primary queue to run all twelve evaluations.

## Connection restored, 2026-09-17

At the user's status request, pan.local became reachable again. The same trainer, orchestrator, conda wrapper, and sleep guard remain alive after roughly 33.4 hours. The smallest student has completed 22 epochs and is training epoch 23. Best completed validation loss is 0.836636 at epoch 22; all recorded loss values are finite, and no training errors were found. No model or full evaluation has completed yet.

Training continued through the earlier hostname-resolution outage without a restart. Current metadata, logs, and histories have been collected locally; no incomplete checkpoints were copied. See local `monitoring.json` for the latest observed batch and time. Continue preserving the healthy run and monitoring hourly.

## Fixed protocol

- Teacher: existing compact-vocabulary FP16 export of `nccratliri/whisperseg-animal-vad`, revision `c2d2502d9b51692fadbb9ac13505d0f50655f4ed`, with 32 encoder and 32 decoder layers, width 1280. This is the large animal-VAD counterpart of the base animal-VAD model, not `large-ms`.
- Teacher weights were transferred and SHA-256 verified: `476122ba41bf6d33964c5c3429f68eaf8e4c70d5230b65d5258291672ede0dd9`.
- Students reuse the existing framewise Conformer checkpoints: 3 layers / width 128 / FFN 128; 5 layers / width 128 / FFN 512; 10 layers / width 128 / FFN 512 (the first 5 initialized from the 5-layer checkpoint).
- Historical size labels are `562k`, `1.92m`, and `4.01m`. Actual hybrid encoder counts, including input/output adapters and positions, are **723,712; 2,085,120; 4,008,320**. Total student counts are **5,733,120; 7,094,528; 9,017,728**.
- All new students start with exactly the same one-layer width-512, FFN-2048 Whisper decoder: **5,009,408 parameters**. Initial decoder SHA-256 is recorded and checked in `smoke.json` and `protocol.json`.
- Clarified with the user on 2026-09-18: fixed decoder means constant architecture and size across students. Both the saved base training code and this sweep freeze decoder weights for epochs 1–2, then fine-tune from epoch 3 with initial LR 3e-6. The user asked about this behavior; no protocol change was requested.
- Decoder source is the existing `models/whisperseg-pruned-models/whisperseg-base-animal-vad-6e1d-dense-fp16` export. Teacher and student token-to-ID maps were checked for exact equality.
- Training: all **3,121 recordings** in `scratch/tests/vad-animals/train`; **16,749 training clips / 6,684 validation clips**, matching prior full-training hybrid logs. Seed 0; batch 4; target length 100; up to 50 epochs; patience 5; minimum improvement 1e-4; encoder LR 3e-4; decoder LR 3e-6; decoder frozen for first 2 epochs; ReduceLROnPlateau unchanged.
- The existing hybrid loader holds out a randomly selected leading or trailing 10% per recording. `split.json` records actual boundaries. An older README's 3,714 final-10% validation clips describes the framewise experiment and must not be applied to this hybrid run.
- New objective: ground-truth CE + 0.5 times KL at temperature 2. Hidden cosine loss is zero because widths differ (512 versus 1280). The historical base hybrid used the retrained base 6E+2D teacher and a hidden cosine term. **Do not attribute all differences solely to teacher size.**
- Multispecies test: same **672 recordings / 7,894 events**. Nightingale: established **2 recordings / 7,362 events**, Chimichanga CH1 and Clipper CH1; **no nightingale fine-tuning**, and no nightingale recordings enter student training.
- All six models get fresh evaluation with FP16 MPS inference, 4 beams, 3 trials, batch 1, native sample rates and the same species-specific settings. Metrics: IoU 0.5 F1, 10-ms collar F1, 1-ms frame F1, SER, boundaries, counts, runtime, and MPS memory. Timings share hardware with other work and are not isolated benchmarks.
- Reference base/large are compact-vocabulary exports preserving all encoder/decoder layers. Report their stored parameter counts accordingly, rather than claiming these are the original full-vocabulary download sizes.

## Implementation and checks

The existing standalone trainer in `models/das-conformer-multispecies/train_evaluate_hybrid.py` was reused. Additions check token IDs and finite updates, print batch progress, and evaluate both normal Whisper directories and hybrid checkpoints under identical beam/batch settings. Its memory sampler now exits with a failed process.

The smoke found that both shared predictor paths passed float32 features into FP16 Whisper models. `src/das/whisperseg/model.py` now casts features to the model's dtype. Both paths have a regression test in `tests/test_whisperseg_model.py`. Wider-teacher KL/backprop coverage is in `tests/unit/test_large_distillation.py`.

- 19 focused and integration tests passed in `das-conformer`.
- Three real MPS teacher/student optimization steps passed with finite gradients; warm step times about 0.93 seconds at batch 4.
- Large teacher inference passed on one real multispecies recording: 19 predicted / 19 true events, IoU F1 1.0. This is only a smoke test, not the full comparison.
- `source/` in the remote run contains the frozen trainer, metrics, and repository source used by this job. Do not replace it while training is active.
- `data-manifest.json`, `split.json`, `teacher-sha256.txt`, `smoke.json`, `smoke-evaluation.json`, `protocol.json`, exact command JSONs, and stage logs preserve provenance.

## Collection and completion

The runner writes `comparison.json` and `comparison.md` as evaluations finish. A complete result requires **12 evaluations**, with the expected file/event counts for each dataset. Do not treat a completed training stage or a smoke score as completion of the user's request.

Collect metadata/logs and completed model directories locally with rsync, excluding `mixed-rate-training-data/`, `smoke-data/`, `smoke-eval-data/`, and `__pycache__/`. Copy final checkpoints only after their training stage completes, then compare remote/local SHA-256. Preserve failure logs. Keep large teacher weights in the established remote model directory; they already exist locally.

If training fails, the existing `training-best.ckpt` supports explicit `--checkpoint` continuation from its best epoch, but its optimizer/scheduler/RNG are not saved. Such recovery resets those states and must be disclosed. The orchestration script does not silently resume or discard partial training; do not invoke it over a partial training directory without first resolving recovery.

After collecting all results, update this handoff and the comparison, provide the user the actual accuracy/size/speed tradeoff and nightingale findings, and delete the heartbeat.

## Final collection, 2026-09-24

The middle student stopped at epoch 49 under the configured patience rule, with best validation loss **0.7548543318 at epoch 44**. Its final checkpoints have 201 finite tensors, matching configs, and an FP16 export equal to the FP32 checkpoint converted to FP16. The five-layer Conformer has 7,094,528 total parameters; tokenizer files match the other new students. All 18 completed training artifacts, including both checkpoints, match by SHA-256 across pan.uol, local, and pan.local. The primary `training-owner.json` hash is preserved.

All six models were evaluated on multispecies (672 files / 7,894 events) and nightingale (2 files / 7,362 events). All twelve result files have finite metrics and matching expected counts. The final comparison records accuracy, parameter count, observed evaluation time, and peak MPS memory. Evaluation times share pan.local with unrelated work and are observed timings, not isolated benchmarks. Training also used different PyTorch versions across pan.uol and pan.local; the historical base-distilled model used a different teacher and objective, so teacher size is not the only changed factor.
