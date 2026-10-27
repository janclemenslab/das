import importlib.util
import json
import shlex
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_measurement_excludes_warmup_and_synchronizes(monkeypatch):
    benchmark = load_script("benchmark_whisperseg_aer")
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        reset_peak_memory_stats=lambda: calls.append("reset"),
        memory_allocated=lambda: 100,
        max_memory_allocated=lambda: 300,
        max_memory_reserved=lambda: 500,
    )
    segmenter = SimpleNamespace(segment=lambda *a, **k: calls.append("segment") or {"onset": [0, 1]})
    ticks = iter([10, 12, 20, 24, 30, 33])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(benchmark, "peak_ram_bytes", lambda: 700)
    result = benchmark.measure(segmenter, np.zeros(60), 1, {"num_beams": 4}, 3, SimpleNamespace(cuda=cuda))
    assert result["inference_seconds"] == [2, 4, 3]
    assert result["inference_seconds_median"] == 3
    assert result["audio_seconds_per_second"] == 20
    assert result["real_time_factor"] == 0.05
    assert result["gpu_peak_allocated_bytes"] == 300
    assert result["gpu_peak_reserved_bytes"] == 500
    assert result["process_peak_ram_bytes"] == 700
    assert calls == ["segment", "sync", "reset"] + ["sync", "segment", "sync"] * 3


def test_bundle_contains_only_excerpt_and_required_code(tmp_path):
    launcher = load_script("whisperseg_aer_hf_job")
    source = tmp_path / "original.wav"
    samples = np.column_stack([np.arange(100) / 100, -np.arange(100) / 100]).astype("float32")
    sf.write(source, samples, 10, subtype="FLOAT")
    bundle = launcher.prepare_bundle(source, tmp_path / "prepared", seconds=3, start=2, channel=1)
    excerpt, sr = sf.read(bundle.parent / "audio.wav", dtype="float32")
    np.testing.assert_array_equal(excerpt, samples[20:50, 1])
    assert sr == 10
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
        assert "original.wav" not in names
        assert "src/das/whisperseg/model.py" in names
        assert "scripts/benchmark_whisperseg_aer.py" in names
        assert len(names) == 9
    manifest = json.loads((bundle.parent / "input-manifest.json").read_text())
    assert manifest["audio_seconds"] == 3
    assert manifest["source_channel"] == 1
    assert len(manifest["source_hashes"]) == 5


def test_all_models_continue_after_worker_failure_and_save_progress(tmp_path, monkeypatch):
    import huggingface_hub

    benchmark = load_script("benchmark_whisperseg_aer")
    args = benchmark.build_parser().parse_args(["--audio", str(tmp_path / "audio.wav"),
                                               "--output-dir", str(tmp_path / "results")])
    args.audio.write_bytes(b"test audio")
    args.output_dir.mkdir()
    (args.output_dir / "base-animal-vad.json").write_text('{"status": "ok", "model": "stale"}')
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda repo: SimpleNamespace(sha="fixed")))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "/cache/model.ckpt")
    commands = []

    def worker(command, **kwargs):
        commands.append(command)
        model = command[command.index("--worker") + 1]
        status = "error" if len(commands) == 1 else "ok"
        path = Path(command[command.index("--worker-output") + 1])
        if status == "ok":
            path.write_text(json.dumps({"model": model, "status": status}))
        return SimpleNamespace(returncode=int(status == "error"))

    monkeypatch.setattr(benchmark.subprocess, "run", worker)
    assert benchmark.run_all(args) == 1
    report = json.loads((args.output_dir / "results.json").read_text())
    assert [row["model"] for row in report["models"]] == list(benchmark.MODEL_OPTIONS)
    assert report["model_revision"] == "fixed"
    assert report["models"][0]["status"] == "error"
    assert len(commands) == len(benchmark.MODEL_OPTIONS)
    assert all(command[0] == sys.executable for command in commands)
    assert (args.output_dir / "results.csv").read_text().count("\n") == len(benchmark.MODEL_OPTIONS) + 1


def test_job_uses_private_repo_and_secret_without_saving_token(tmp_path, monkeypatch):
    import huggingface_hub

    launcher = load_script("whisperseg_aer_hf_job")
    calls = {}
    api = SimpleNamespace(
        whoami=lambda: {"name": "tester", "auth": {"accessToken": {"role": "write"}}},
        create_repo=lambda repo, **kw: calls.update(create=(repo, kw)),
        upload_file=lambda **kw: calls.update(upload=kw),
        run_job=lambda **kw: calls.update(job=kw) or SimpleNamespace(id="123", url="https://example/job/123"),
    )
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-secret")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: api)
    launcher.submit(tmp_path / "inputs.tar.gz", tmp_path)
    assert calls["create"][1] == {"repo_type": "dataset", "private": True}
    assert calls["job"]["flavor"] == "t4-small"
    assert calls["job"]["timeout"] == "1h"
    assert calls["job"]["secrets"] == {"HF_TOKEN": "test-secret"}
    assert "test-secret" not in (tmp_path / "job.json").read_text()
    shell = calls["job"]["command"][2]
    bootstrap = shlex.split(shell.rsplit("python -c ", 1)[1])[0]
    compile(bootstrap, "bootstrap", "exec")


def test_read_only_token_fails_before_upload(tmp_path, monkeypatch):
    import huggingface_hub

    launcher = load_script("whisperseg_aer_hf_job")
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-secret")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(
        whoami=lambda: {"auth": {"accessToken": {"role": "read"}}},
    ))
    with pytest.raises(RuntimeError, match="read-only"):
        launcher.submit(tmp_path / "inputs.tar.gz", tmp_path)


def test_manifest_runs_only_selected_models_with_pinned_sources(tmp_path, monkeypatch):
    import huggingface_hub

    benchmark = load_script("benchmark_whisperseg_aer")
    specs = [
        {"model": "tiny-compact", "path": "models/tiny", "format": "transformers", "dtype": "float16"},
        {"model": "tiny-control", "repo": "speech/tiny", "revision": "fixed", "format": "transformers",
         "dtype": "float32", "control_template": {"repo": "segment/base", "revision": "template-pin"}},
    ]
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps(specs))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    args = benchmark.build_parser().parse_args(["--audio", str(audio), "--models", str(manifest),
                                               "--output-dir", str(tmp_path / "results")])
    downloads, commands = [], []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw:
                        downloads.append((repo, kw)) or "/cache/" + repo)

    def worker(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--worker-output") + 1]).write_text(json.dumps({"status": "ok"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", worker)
    assert benchmark.run_all(args) == 0
    assert len(commands) == 2
    assert commands[0][commands[0].index("--checkpoint") + 1] == str(tmp_path / "models/tiny")
    assert commands[1][commands[1].index("--dtype") + 1] == "float32"
    assert "--control-template" in commands[1]
    assert [kw["revision"] for _, kw in downloads] == ["fixed", "template-pin"]
    report = json.loads((args.output_dir / "results.json").read_text())
    assert [row["source"] for row in report["models"]] == specs


def test_completed_resource_benchmark_is_reused_and_changed_audio_rejected(tmp_path, monkeypatch):
    import hashlib

    benchmark = load_script('benchmark_whisperseg_aer')
    spec = {'model': 'tiny', 'path': 'tiny', 'format': 'transformers', 'dtype': 'float32'}
    manifest = tmp_path / 'models.json'
    manifest.write_text(json.dumps([spec]))
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    args = benchmark.build_parser().parse_args(['--audio', str(audio), '--models', str(manifest),
        '--output-dir', str(tmp_path / 'results'), '--device', 'mps'])
    args.output_dir.mkdir()
    row = {'status': 'ok', 'source': spec, 'device': 'mps', 'platform': benchmark.platform.platform(),
        'cpu_threads': 4, 'channel': 0, 'inference_seconds': [1] * 5,
        'settings': {k: getattr(args, k) for k in ('batch_size', 'num_beams', 'num_trials',
                    'max_length', 'spec_time_step', 'min_frequency')}}
    result = args.output_dir / 'tiny.json'
    result.write_text(json.dumps(row))
    original = result.read_bytes()
    (args.output_dir / 'results.json').write_text(json.dumps({'audio_sha256': hashlib.sha256(b'audio').hexdigest()}))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **kw: pytest.fail('Repeated completed work'))
    assert benchmark.run_all(args) == 0
    assert result.read_bytes() == original
    audio.write_bytes(b'changed')
    with pytest.raises(AssertionError):
        benchmark.run_all(args)


def test_bundle_includes_selected_local_checkpoint_hashes(tmp_path):
    launcher = load_script("whisperseg_aer_hf_job")
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(10), 10)
    directory = tmp_path / "models/tiny"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"weights")
    models = tmp_path / "models.json"
    models.write_text(json.dumps([{"path": "models/tiny"}, {"path": "models/tiny"}]))
    bundle = launcher.prepare_bundle(audio, tmp_path / "prepared", 1, 0, 0, models)
    with tarfile.open(bundle) as archive:
        assert archive.getnames().count("models/tiny/model.safetensors") == 1
        assert archive.extractfile("models/tiny/model.safetensors").read() == b"weights"
        manifest = json.load(archive.extractfile("input-manifest.json"))
        assert len(manifest["asset_hashes"]["models/tiny/model.safetensors"]) == 64
    models.write_text(json.dumps([{"path": "../outside"}]))
    with pytest.raises(AssertionError):
        launcher.prepare_bundle(audio, tmp_path / "invalid", 1, 0, 0, models)


def test_full_vocabulary_control_crops_encoder_without_compacting(tmp_path, monkeypatch):
    import torch
    import transformers

    benchmark = load_script("benchmark_whisperseg_aer")
    config = transformers.WhisperConfig(vocab_size=16, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=1, decoder_attention_heads=1, encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_source_positions=1500, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = transformers.WhisperForConditionalGeneration(config)
    template = transformers.WhisperConfig(**config.to_dict())
    template.max_source_positions = 500
    template.total_spec_columns = 1000
    template.cluster_codebook = {"vocal": 0}
    template.species_codebook = {}
    template.forced_decoder_ids = None
    before = model.model.encoder.embed_positions.weight[:500].detach().clone()
    monkeypatch.setattr(transformers.WhisperForConditionalGeneration, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(transformers.WhisperConfig, "from_pretrained", lambda *a, **kw: template)
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: transformers.GenerationConfig())
    monkeypatch.setattr(transformers.WhisperTokenizer, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(get_vocab=lambda: {"token": 15}))
    monkeypatch.setattr(model, "to", lambda *a, **kw: model)  # Keep this loading check on the CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100))
    monkeypatch.setattr(benchmark, "measure", lambda *a: {})
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(10), 10)
    args = benchmark.build_parser().parse_args(["--audio", str(audio), "--worker", "tiny-control",
        "--checkpoint", "speech-source", "--format", "transformers", "--dtype", "float32",
        "--control-template", "template", "--worker-output", str(tmp_path / "result.json")])
    assert benchmark.run_worker(args) == 0
    assert model.config.vocab_size == 16
    assert model.config.max_source_positions == model.model.encoder.embed_positions.num_embeddings == 500
    torch.testing.assert_close(model.model.encoder.embed_positions.weight, before)
    result = json.loads(args.worker_output.read_text())
    assert result["architecture_control"] is True
    assert result["dtype"] == "torch.float32"


def test_mps_measurement_tracks_memory_and_excludes_warmup_tokens(monkeypatch):
    benchmark = load_script("benchmark_whisperseg_aer")
    trace, calls = [], []
    mps = SimpleNamespace(synchronize=lambda: calls.append("sync"), current_allocated_memory=lambda: 100,
                          driver_allocated_memory=lambda: 200)

    def segment(*a, **k):
        trace.append({"tokens": 12, "prefix_tokens": 4, "batch": 1})
        return {"onset": [0]}

    segmenter = SimpleNamespace(segment=segment, generation_trace=trace)
    ticks = iter([10, 12, 20, 24])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    result = benchmark.measure(segmenter, np.zeros(60), 1, {}, 2, SimpleNamespace(mps=mps), "mps")
    assert result["inference_seconds"] == [2, 4]
    assert result["mps_peak_current_bytes"] == 100 and result["mps_peak_driver_bytes"] == 200
    assert "gpu_peak_allocated_bytes" not in result
    assert result["generation_traces"] == [[{"tokens": 12, "prefix_tokens": 4, "batch": 1}]] * 2
