"""Run the large-teacher sweep using a frozen copy of the existing experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


VARIANTS = (("562k", "framewise-562k", 3), ("1.92m", "framewise-1.92m", 5), ("4.01m", "framewise-1.92m", 10))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def locations(base):
    models = base / "models"
    return (
        models / "das-conformer-multispecies",
        models / "whisperseg-pruned-models/whisperseg-base-animal-vad-6e1d-dense-fp16",
        models / "whisperseg-compact-models/whisperseg-animal-vad-compact-vocab-fp16",
    )


def smoke(root, base):
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer
    import train_evaluate_hybrid as hybrid

    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    frames, decoder, teacher_path = locations(base)
    tokenizer = WhisperTokenizer.from_pretrained(decoder, language="english")
    teacher_tokenizer = WhisperTokenizer.from_pretrained(teacher_path, language="english")
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    np.random.seed(0)
    torch.manual_seed(0)
    student = hybrid.build_hybrid(decoder, frames / "framewise-562k/best-model.ckpt")
    data = hybrid.prepare_mixed_rate_data(base / "scratch/tests/vad-animals/train", root / "smoke-data")
    training, validation, n_train, n_val = hybrid.make_loaders(data, student, tokenizer, 4, 0, 100)
    # Record actual split boundaries; the loader randomly chooses the leading or trailing 10%.
    split = {}
    for name, loader in (("train", training), ("validation", validation)):
        split[name] = sorted({(Path(r.audio_path).name, r.interval_start_sample, r.interval_stop_sample) for r in loader.dataset.records})
    write_json(root / "split.json", split)
    batch = next(iter(training))
    teacher = WhisperForConditionalGeneration.from_pretrained(teacher_path, dtype=torch.float16).to("mps").eval()
    student.to("mps").train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    timings = []
    losses = []
    for _ in range(3):
        started = time.perf_counter()
        optimizer.zero_grad()
        loss, ce, kl, hidden = hybrid.distillation_loss(student, teacher, batch, torch.device("mps"))
        assert torch.isfinite(loss) and hidden == 0
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        timings.append(time.perf_counter() - started)
        losses.append(dict(loss=float(loss.detach()), ce=float(ce.detach()), kl=float(kl.detach()), gradient_norm=float(norm)))
        print({"seconds": timings[-1], **losses[-1]}, flush=True)
    del teacher, student, optimizer, loss, ce, kl, hidden
    import gc
    gc.collect()
    torch.mps.empty_cache()
    variants = []
    decoder_hash = None
    for name, frame, layers in VARIANTS:
        torch.manual_seed(0)
        model = hybrid.build_hybrid(decoder, frames / frame / "best-model.ckpt", layers)
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            if not key.startswith("model.encoder."):
                digest.update(key.encode())
                digest.update(value.numpy().tobytes())
        current_hash = digest.hexdigest()
        decoder_hash = decoder_hash or current_hash
        assert current_hash == decoder_hash, "decoder initialization must match across sizes"
        variants.append({"name": name, "parameters": sum(p.numel() for p in model.parameters()), "encoder_parameters": sum(p.numel() for p in model.model.encoder.parameters()), "decoder_sha256": current_hash})
        del model
    result = {"passed": True, "device": "mps", "torch": torch.__version__, "training_clips": n_train, "validation_clips": n_val, "batch_size": 4, "seconds_per_step": timings, "losses": losses, "variants": variants, "estimated_training_epoch_seconds": timings[-1] * len(training)}
    write_json(root / "smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)


def report(root):
    rows = []
    for path in sorted(root.glob("*/evaluation-*.json")):
        result = json.loads(path.read_text())
        dataset = path.stem.removeprefix("evaluation-")
        assert result["protocol"]["files"] == {"multispecies": 672, "nightingale": 2}[dataset]
        assert result["metrics"]["truth_events"] == {"multispecies": 7894, "nightingale": 7362}[dataset]
        rows.append({"model": path.parent.name, "dataset": dataset, "parameters": result["parameters"], "seconds": result["elapsed_seconds"], "peak_mps_mib": result["peak_mps_driver_bytes"] / 2**20, **result["metrics"]})
    write_json(root / "comparison.json", rows)
    lines = ["# WhisperSeg large distillation", "", "Status: " + ("complete" if len(rows) == 12 else f"in progress ({len(rows)}/12 evaluations)"), "", "All rows use the same MPS device, FP16 inference, four beams, three trials, and batch size one. Nightingale is evaluated without fine-tuning. Training uses the original 3,121 recordings; multispecies test has 672 recordings. The new students share the same initial one-layer, width-512, FFN-2048 Whisper decoder. Only Conformer capacity varies.", "", "The large teacher is the animal-VAD variant (32 encoder and 32 decoder layers), stored with the existing compact vocabulary. The historical base student used the retrained base 6E+2D teacher. Its objective included hidden-state cosine loss; large-to-width-512 distillation uses CE + 0.5 KL at temperature 2 because hidden widths differ (1280 versus 512). This comparison therefore does not isolate teacher size alone. Parameter counts describe compact-vocabulary models, not the original full-vocabulary downloads.", "", "Validation follows the existing seed-0 loader: a leading or trailing 10% of each training recording, with boundaries saved in split.json. Test data is never used for training or model selection. The two nightingale recordings are the established Chimichanga CH1 and Clipper CH1 test set; no nightingale recordings enter this sweep's training.", "", "| Model | Test set | Parameters | IoU F1 | Collar F1 | Frame F1 | SER | Eval seconds | Peak MPS MiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['parameters']:,} | {row['iou_f1']:.4f} | {row['collar_f1']:.4f} | {row['frame_f1']:.4f} | {row['syllable_error_rate']:.4f} | {row['seconds']:.1f} | {row['peak_mps_mib']:.0f} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def wait_for_owned_training(root, output, target):
    owner_path = output / "training-owner.json"
    if not owner_path.exists():
        return
    owner = json.loads(owner_path.read_text())
    status = {"model": output.name, "stage": "train", "pid": os.getpid(), "owner": owner}
    if "local_pid" in owner:
        if target.exists():
            json.loads(target.read_text())
            return
        write_json(root / "status.json", {**status, "state": "running", "child_pid": owner["local_pid"], "adopted": True})
        while True:
            process = subprocess.run(["ps", "-p", str(owner["local_pid"]), "-o", "command="], capture_output=True, text=True)
            if not process.stdout.strip():
                break
            if str(output) not in process.stdout or "train_evaluate_hybrid.py" not in process.stdout:
                raise RuntimeError("Adopted training PID no longer matches its recorded command")
            time.sleep(30)
    else:
        write_json(root / "status.json", {**status, "state": "waiting_external"})
        # Only the collector writes this after checkpoint transfer and SHA-256 verification.
        while not (output / "transfer-verified.json").exists():
            time.sleep(30)
    if not target.exists():
        raise RuntimeError(f"Owned training did not finish successfully: {output}")
    json.loads(target.read_text())


def run(root, base):
    frames, decoder, teacher = locations(base)
    assert json.loads((root / "smoke.json").read_text())["passed"]
    trainer = root / "source/train_evaluate_hybrid.py"
    env = {**os.environ, "PYTHONPATH": str(root / "source/src"), "PYTORCH_ENABLE_MPS_FALLBACK": "0", "OMP_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4"}
    stages = []
    for name, frame, layers in VARIANTS:
        output = root / f"large-distilled-{name}"
        command = ["--train-dir", base / "scratch/tests/vad-animals/train", "--output-dir", output, "--decoder-source", decoder, "--frame-checkpoint", frames / frame / "best-model.ckpt", "--teacher", teacher, "--encoder-layers", layers, "--device", "mps", "--batch-size", 4, "--num-workers", 4, "--target-length", 100, "--learning-rate", "0.0003", "--decoder-learning-rate", "0.000003", "--freeze-decoder-epochs", 2, "--max-epochs", 50, "--patience", 5, "--seed", 0]
        stages.append((output, "train", command, output / "training_result.json"))
    checkpoints = [(f"large-distilled-{name}", root / f"large-distilled-{name}/best-model.ckpt") for name, _, _ in VARIANTS]
    checkpoints += [("whisperseg-base", base / "models/whisperseg-compact-models/whisperseg-base-animal-vad-compact-vocab-fp16"), ("whisperseg-large", teacher), ("base-distilled-1.92m", frames / "hybrid-1.92m-whisper-1d/best-model.ckpt")]
    for name, checkpoint in checkpoints:
        output = root / name
        for dataset, data in (("multispecies", "vad-animals/test"), ("nightingale", "nightingale-test")):
            target = output / f"evaluation-{dataset}.json"
            stages.append((output, dataset, ["--evaluate-only", "--checkpoint", checkpoint, "--output-dir", output, "--test-dir", base / "scratch/tests" / data, "--evaluation-output", target, "--device", "mps", "--num-trials", 3, "--num-beams", 4, "--evaluation-batch-size", 1], target))
    write_json(root / "commands.json", [[str(x) for x in stage[2]] for stage in stages])
    for output, stage, arguments, target in stages:
        if stage == "train":
            wait_for_owned_training(root, output, target)
        if target.exists():
            json.loads(target.read_text())
            continue
        output.mkdir(exist_ok=True)
        status = {"state": "running", "model": output.name, "stage": stage, "pid": os.getpid(), "started": time.time()}
        write_json(root / "status.json", status)
        command = [sys.executable, "-u", str(trainer), *map(str, arguments)]
        write_json(output / f"{stage}-command.json", command)
        with (output / f"{stage}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(root / "status.json", {**status, "child_pid": child.pid})
            code = child.wait()
        if code or not target.exists():
            write_json(root / "status.json", {**status, "state": "failed", "exit_code": code})
            raise RuntimeError(f"{output.name} {stage} failed; see {output / (stage + '.log')}")
        report(root)
    report(root)
    assert len(json.loads((root / "comparison.json").read_text())) == 12
    write_json(root / "status.json", {"state": "complete", "finished": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("/Users/janc/segmentation-refits"))
    parser.add_argument("--phase", choices=("smoke", "run", "report"), required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root / "source"))
    if args.phase == "report":
        report(args.root)
    else:
        globals()[args.phase](args.root, args.base)


if __name__ == "__main__":
    main()
