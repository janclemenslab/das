"""Benchmark WhisperSeg checkpoints in a fresh CUDA or MPS process."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from das.whisperseg.catalog import MODEL_OPTIONS, MODEL_REPO


def peak_ram_bytes():
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure(segmenter, audio, sr, settings, repeats, torch, device="cuda"):
    # Warm up the entire pipeline, including lazy feature extraction/JIT setup.
    segmenter.segment(audio, sr=sr, **settings)
    accelerator = getattr(torch, device)
    accelerator.synchronize()
    memory = {}
    stop = threading.Event()
    if device == "cuda":
        accelerator.reset_peak_memory_stats()
        memory["gpu_model_bytes"] = accelerator.memory_allocated()
    else:
        memory.update(mps_model_bytes=accelerator.current_allocated_memory(),
                      mps_peak_current_bytes=accelerator.current_allocated_memory(),
                      mps_peak_driver_bytes=accelerator.driver_allocated_memory())

        def sample_memory():
            while not stop.wait(0.01):
                memory["mps_peak_current_bytes"] = max(memory["mps_peak_current_bytes"], accelerator.current_allocated_memory())
                memory["mps_peak_driver_bytes"] = max(memory["mps_peak_driver_bytes"], accelerator.driver_allocated_memory())

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
    seconds, event_counts, generation_traces = [], [], []
    try:
        for _ in range(repeats):
            trace = getattr(segmenter, "generation_trace", [])
            trace.clear()
            accelerator.synchronize()
            started = time.perf_counter()
            prediction = segmenter.segment(audio, sr=sr, **settings)
            accelerator.synchronize()
            seconds.append(time.perf_counter() - started)
            event_counts.append(len(prediction["onset"]))
            generation_traces.append(list(trace))
    finally:
        stop.set()
        if device == "mps":
            sampler.join()
            memory["mps_peak_current_bytes"] = max(memory["mps_peak_current_bytes"], accelerator.current_allocated_memory())
            memory["mps_peak_driver_bytes"] = max(memory["mps_peak_driver_bytes"], accelerator.driver_allocated_memory())
    if device == "cuda":
        memory.update(gpu_peak_allocated_bytes=accelerator.max_memory_allocated(),
                      gpu_peak_reserved_bytes=accelerator.max_memory_reserved())
    median = statistics.median(seconds)
    duration = len(audio) / sr
    return {
        "audio_seconds": duration,
        "inference_seconds": seconds,
        "inference_seconds_median": median,
        "real_time_factor": median / duration,
        "audio_seconds_per_second": duration / median,
        "event_counts": event_counts,
        "generation_traces": generation_traces,
        **memory,
        "process_peak_ram_bytes": peak_ram_bytes(),
    }


def run_worker(args):
    row = {"model": args.worker, "status": "error"}
    try:
        import soundfile as sf
        import torch
        import transformers
        from das.whisperseg.model import WhisperSegmenterForEval

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run this benchmark on an NVIDIA GPU.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required; run this benchmark on an Apple GPU.")
        torch.manual_seed(0)
        torch.set_num_threads(args.cpu_threads)
        audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
        audio = audio[:, args.channel].copy()
        dtype = getattr(torch, args.dtype)
        if args.format == "transformers":
            from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperTokenizer

            model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint, dtype=dtype)
            tokenizer_path = args.control_template or args.checkpoint
            tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path, language="english")
            if args.control_template:
                # Speech weights provide an architecture control, not a trained segmenter.
                template = WhisperConfig.from_pretrained(args.control_template)
                for key in ("total_spec_columns", "max_source_positions", "cluster_codebook", "species_codebook",
                            "pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id",
                            "forced_decoder_ids", "suppress_tokens", "begin_suppress_tokens"):
                    setattr(model.config, key, getattr(template, key))
                assert model.config.vocab_size == template.vocab_size
                positions = model.model.encoder.embed_positions
                positions.weight = torch.nn.Parameter(positions.weight[:template.max_source_positions].clone())
                positions.num_embeddings = template.max_source_positions
                model.generation_config = GenerationConfig.from_pretrained(args.control_template)
            assert max(tokenizer.get_vocab().values()) < model.config.vocab_size
            segmenter = WhisperSegmenterForEval(model=model.to(args.device), tokenizer=tokenizer)
        else:
            segmenter = WhisperSegmenterForEval(args.checkpoint, device=args.device)
        segmenter.model.to(dtype=dtype).eval()
        segmenter.generation_trace = []
        generate = segmenter.model.generate

        def traced_generate(*call_args, **call_kwargs):
            ids = generate(*call_args, **call_kwargs)
            prefix = call_kwargs["decoder_input_ids"].shape[-1]
            segmenter.generation_trace.append({"batch": int(ids.shape[0]), "tokens": int(ids.shape[-1]),
                                               "prefix_tokens": int(prefix)})
            return ids

        segmenter.model.generate = traced_generate
        settings = {key: getattr(args, key) for key in (
            "batch_size", "num_beams", "num_trials", "max_length", "spec_time_step", "min_frequency"
        )}
        with torch.inference_mode():
            row.update(measure(segmenter, audio, sr, settings, args.repeats, torch, args.device))
        row.update(
            status="ok", settings=settings, dtype=str(next(segmenter.model.parameters()).dtype),
            parameters=sum(p.numel() for p in segmenter.model.parameters()),
            device=args.device, platform=platform.platform(),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            transformers_version=transformers.__version__, python_version=platform.python_version(),
            cpu_threads=args.cpu_threads, sample_rate=sr, channel=args.channel,
            vocabulary_size=segmenter.model.config.vocab_size, checkpoint_format=args.format,
            architecture_control=bool(args.control_template),
        )
        if args.device == "cuda":
            row.update(gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        else:
            row.update(hardware=subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip(),
                       memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)))
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc), process_peak_ram_bytes=peak_ram_bytes())
    args.worker_output.write_text(json.dumps(row, indent=2) + "\n")
    return 0 if row["status"] == "ok" else 1


def save_results(report, output_dir, results_repo=None, results_prefix="results"):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "model", "status", "dtype", "vocabulary_size", "parameters", "architecture_control",
        "audio_seconds", "inference_seconds_median", "real_time_factor",
        "audio_seconds_per_second", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
        "device", "mps_peak_current_bytes", "mps_peak_driver_bytes",
        "process_peak_ram_bytes", "error_type", "error",
    ]
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["models"])
    if results_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=results_repo, repo_type="dataset", folder_path=output_dir,
            path_in_repo=results_prefix, commit_message="Save CUDA benchmark progress",
        )


def run_all(args):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = None if args.models else HfApi().model_info(MODEL_REPO).sha
    models = json.loads(args.models.read_text()) if args.models else [
        {"model": alias, "repo": MODEL_REPO, "revision": revision, "filename": filename,
         "format": "checkpoint", "dtype": "float16"}
        for alias, (_, filename) in MODEL_OPTIONS.items()
    ]
    assert len({spec["model"] for spec in models}) == len(models), "Model names must be unique."
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(), "model_repo": None if args.models else MODEL_REPO,
        "model_revision": revision, "audio_file": args.audio.name,
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "protocol": "One full warmup, then repeated segmentation of in-memory audio; fresh process per model.",
        "memory_scope": "CUDA: PyTorch allocated/reserved peaks during repeats, including resident model. "
                        "MPS: current/driver allocation sampled every 10 ms after warmup, including resident model. "
                        "RAM: process RSS high-water mark, including imports, audio and model loading.",
        "models": [],
    }
    manifest = args.audio.parent / "input-manifest.json"
    if manifest.exists():
        report["input_manifest"] = json.loads(manifest.read_text())
    for spec in models:
        alias = spec["model"]
        name = alias.split(":", 1)[-1]
        assert Path(name).name == name and name not in (".", ".."), "Unsafe model name."
        result_path = args.output_dir / f"{name}.json"
        if result_path.exists():
            row = json.loads(result_path.read_text())
            if row['status'] == 'ok' and 'source' in row:
                previous = json.loads((args.output_dir / 'results.json').read_text())
                assert previous['audio_sha256'] == report['audio_sha256']
                assert row['source'] == spec and row['device'] == args.device
                assert row['platform'] == platform.platform()
                assert row['cpu_threads'] == args.cpu_threads and row['channel'] == args.channel
                assert len(row['inference_seconds']) == args.repeats
                assert row['settings'] == {key: getattr(args, key) for key in (
                    'batch_size', 'num_beams', 'num_trials', 'max_length', 'spec_time_step', 'min_frequency')}
                report['models'].append(row)
                save_results(report, args.output_dir, args.results_repo, args.results_prefix)
                print(f'Reusing completed benchmark {alias}', flush=True)
                continue
        print(f"Benchmarking {alias}", flush=True)
        try:
            if "path" in spec:
                checkpoint = str((args.models.parent / spec["path"]).resolve())
            elif spec["format"] == "checkpoint":
                checkpoint = hf_hub_download(spec["repo"], spec["filename"], revision=spec["revision"])
            else:
                checkpoint = snapshot_download(spec["repo"], revision=spec["revision"],
                    allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model*.bin"])
            command = [
                sys.executable, str(Path(__file__).resolve()), "--audio", str(args.audio.resolve()),
                "--worker", alias, "--checkpoint", checkpoint, "--worker-output", str(result_path.resolve()),
                "--format", spec["format"], "--dtype", spec["dtype"],
                "--device", args.device,
            ]
            if spec.get("control_template"):
                template = spec["control_template"]
                template_path = snapshot_download(template["repo"], revision=template["revision"],
                                                 allow_patterns=["*.json", "*.txt"])
                command.extend(["--control-template", template_path])
            for key in ("batch_size", "num_beams", "num_trials", "max_length", "repeats", "channel",
                        "spec_time_step", "min_frequency", "cpu_threads"):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            result_path.unlink(missing_ok=True)
            completed = subprocess.run(command, check=False)
            if result_path.exists():
                row = json.loads(result_path.read_text())
                if completed.returncode and row["status"] == "ok":
                    row.update(status="error", error=f"Worker exited {completed.returncode}")
            else:
                row = {"model": alias, "status": "error", "error": f"Worker exited {completed.returncode}"}
        except Exception as exc:
            row = {"model": alias, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        row["source"] = spec
        report["models"].append(row)
        print(json.dumps(row), flush=True)
        save_results(report, args.output_dir, args.results_repo, args.results_prefix)
    return int(any(row["status"] != "ok" for row in report["models"]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="WAV/FLAC file; benchmark uses the full file")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--spec-time-step", type=float, default=0.0025)
    parser.add_argument("--min-frequency", type=float, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    parser.add_argument("--results-repo", help="Existing private Hub dataset repository for incremental results")
    parser.add_argument("--results-prefix", default="results")
    parser.add_argument("--models", type=Path, help="JSON list of pinned checkpoints; default: all seven AER releases")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--format", choices=("checkpoint", "transformers"), default="checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16", help=argparse.SUPPRESS)
    parser.add_argument("--control-template", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or args.num_trials < 1 or args.num_beams < 1 or args.cpu_threads < 1:
        parser.error("Repeats, batch size, trials, beams and CPU threads must be positive.")
    if args.worker and (not args.checkpoint or not args.worker_output):
        parser.error("Worker requires --checkpoint and --worker-output.")
    sys.exit(run_worker(args) if args.worker else run_all(args))
