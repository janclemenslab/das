"""Prepare or submit the T4 benchmark, keeping audio and results in a private Hub repo."""

import argparse
import hashlib
import io
import json
import shlex
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = Path("/Users/janc/Library/CloudStorage/Dropbox/birds/ng_match/dat.raw/"
                     "Chimichanga_NW/BOSCON_BOS_Chimichanga_03052021_denoisedCH1.wav")
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"


def prepare_bundle(audio_path, output_dir, seconds, start, channel, models=None):
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(audio_path) as source:
        sr = source.samplerate
        source.seek(round(start * sr))
        audio = source.read(round(seconds * sr), dtype="float32", always_2d=True)
    if not len(audio):
        raise ValueError("Selected audio excerpt is empty.")
    excerpt = output_dir / "audio.wav"
    sf.write(excerpt, audio[:, channel], sr, subtype="FLOAT")
    sources = [ROOT / "scripts/benchmark_whisperseg_aer.py"]
    sources += [ROOT / "src/das/whisperseg" / name for name in ("catalog.py", "model.py", "audio_utils.py", "utils.py")]
    assets = []
    seen_assets = set()
    if models:
        assets.append((models, "models.json"))
        for spec in json.loads(models.read_text()):
            if "path" in spec:
                assert not Path(spec["path"]).is_absolute() and ".." not in Path(spec["path"]).parts
                directory = models.parent / spec["path"]
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        name = (Path(spec["path"]) / path.relative_to(directory)).as_posix()
                        if name not in seen_assets:
                            assets.append((path, name))
                            seen_assets.add(name)
    manifest = {
        "source_audio": str(audio_path.resolve()),
        "source_audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "excerpt_sha256": hashlib.sha256(excerpt.read_bytes()).hexdigest(),
        "start_seconds": start, "audio_seconds": len(audio) / sr, "sample_rate": sr, "source_channel": channel,
        "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "asset_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in assets},
    }
    manifest_path = output_dir / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    bundle = output_dir / "inputs.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=1) as archive:
        archive.add(excerpt, arcname="audio.wav")
        archive.add(manifest_path, arcname="input-manifest.json")
        for path in sources:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
        for path, name in assets:
            archive.add(path, arcname=name)
        for name in ("src/das/__init__.py", "src/das/whisperseg/__init__.py"):
            archive.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
    return bundle


def job_command(repo_id):
    bootstrap = f"""from pathlib import Path
import os, subprocess, sys, tarfile
from huggingface_hub import hf_hub_download
root = Path('/tmp/whisperseg-aer-benchmark')
root.mkdir(exist_ok=True)
with tarfile.open(hf_hub_download({repo_id!r}, 'inputs.tar.gz', repo_type='dataset')) as archive:
    archive.extractall(root, filter='data')
os.chdir(root)
raise SystemExit(subprocess.call([sys.executable, 'scripts/benchmark_whisperseg_aer.py',
    '--audio', 'audio.wav', '--output-dir', 'results', '--results-repo', {repo_id!r}]
    + (['--models', 'models.json'] if Path('models.json').exists() else [])))
"""
    install = " ".join(shlex.quote(package) for package in (
        "transformers==4.57.1", "numpy<3", "scipy", "scikit-learn", "soundfile", "huggingface_hub==0.36.2",
    ))
    command = (
        "conda create -y -n das-conformer --clone base && "
        "conda run -n das-conformer python -m pip install uv && "
        f"conda run -n das-conformer uv pip install --python /opt/conda/envs/das-conformer/bin/python {install} && "
        f"conda run --no-capture-output -n das-conformer python -c {shlex.quote(bootstrap)}"
    )
    return ["bash", "-lc", command]


def submit(bundle, output_dir):
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise RuntimeError("Run hf auth login with a token that can write repositories and launch Jobs.")
    api = HfApi(token=token)
    identity = api.whoami()
    if identity.get("auth", {}).get("accessToken", {}).get("role") == "read":
        raise RuntimeError("Saved token is read-only. Run hf auth login with repository write and Jobs permissions.")
    repo_id = f"{identity['name']}/whisperseg-aer-benchmark-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}"
    api.create_repo(repo_id, repo_type="dataset", private=True)
    api.upload_file(path_or_fileobj=bundle, path_in_repo="inputs.tar.gz", repo_id=repo_id, repo_type="dataset")
    job = api.run_job(
        image=IMAGE, command=job_command(repo_id), flavor="t4-small", timeout="1h",
        secrets={"HF_TOKEN": token}, env={"OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"},
    )
    details = {"job_id": job.id, "job_url": job.url, "results_repo": repo_id,
               "results_url": f"https://huggingface.co/datasets/{repo_id}/tree/main/results",
               "flavor": "t4-small", "timeout": "1h", "image": IMAGE}
    (output_dir / "job.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/whisperseg-aer-cuda")
    parser.add_argument("--submit", action="store_true", help="Upload private inputs and launch the paid T4 job")
    parser.add_argument("--models", type=Path, help="JSON model manifest, including optional local model directories")
    args = parser.parse_args()
    if args.seconds <= 0 or args.start < 0 or args.channel < 0:
        parser.error("Seconds must be positive; start and channel must be nonnegative.")
    bundle = prepare_bundle(args.audio, args.output_dir, args.seconds, args.start, args.channel, args.models)
    print(f"Prepared {bundle}", flush=True)
    if args.submit:
        submit(bundle, args.output_dir)
