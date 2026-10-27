"""Evaluate our Animal2vec export on the existing nightingale split, on MPS."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

import nightingale_encoder_benchmark as benchmark


class Segmenter(torch.nn.Module):
    def __init__(self, encoder, weights):
        super().__init__()
        self.encoder = encoder
        self.register_buffer('weights', weights.detach())

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return (self.encoder(waveform) @ self.weights).sigmoid()


def extract(wav, encoder, metadata, output):
    started = time.monotonic()
    annotation = json.loads(wav.with_suffix('.json').read_text())
    assert set(annotation['cluster']) == {'vocal'}
    truth = list(zip(annotation['onset'], annotation['offset'], strict=True))
    info = sf.info(wav)
    assert info.samplerate == 44100 and info.channels == 1
    window_seconds = metadata['window_seconds']
    frames_per_window, dimension = metadata['output_shape'][1:]
    hop = window_seconds / frames_per_window
    frames = int(np.ceil(info.duration / hop))
    window_samples = round(window_seconds * info.samplerate)
    cache = output / 'features-mps' / (wav.stem + '.npy')
    if not cache.exists():
        temporary = cache.with_suffix('.tmp.npy')
        features = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(frames, dimension))
        with sf.SoundFile(wav) as audio:
            for index, start in enumerate(range(0, info.frames, window_samples)):
                chunk = resample_poly(audio.read(window_samples, dtype='float32'), 80, 147)
                chunk = np.pad(chunk[:60000], (0, max(0, 60000 - len(chunk))))
                with torch.no_grad():
                    block = encoder(torch.from_numpy(chunk[None]).to('mps'))
                assert block.device.type == 'mps' and torch.isfinite(block).all()
                first = index * frames_per_window
                count = min(frames_per_window, frames - first)
                features[first:first + count] = block[0, :count].cpu().numpy()
                if index % 50 == 0 or start + window_samples >= info.frames:
                    status = dict(stage='embedding', device='mps', recording=wav.name,
                                  windows=index + 1, total_windows=int(np.ceil(info.frames / window_samples)),
                                  elapsed_seconds=time.monotonic() - started)
                    benchmark.save_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        features.flush()
        del features
        temporary.replace(cache)
    features = np.load(cache, mmap_mode='r')
    assert features.shape == (frames, dimension)
    starts = np.arange(frames) // frames_per_window * window_seconds
    boundary = 0.9 * info.duration
    validation_start = np.ceil(boundary / window_seconds) * window_seconds
    centers = np.minimum((np.arange(frames) + 0.5) * hop, info.duration - 1e-6)
    return dict(name=wav.name, features=features, truth=truth, hop=hop, duration=info.duration,
                labels=benchmark.labels_at(centers, truth), train_mask=starts + window_seconds <= boundary,
                validation_mask=starts >= validation_start, validation_start=float(validation_start),
                extraction_seconds=time.monotonic() - started)


def main(root, reference_root):
    assert sys.version_info[:2] == (3, 14), 'Verify in the target Python 3.14 runtime'
    assert torch.backends.mps.is_available(), 'MPS GPU is required'
    torch.set_num_threads(4)
    # Per-operation profiler synchronization crashes TorchScript loading in torch 2.13.
    started = time.monotonic()
    output = root / 'evaluation'
    (output / 'features-mps').mkdir(parents=True, exist_ok=True)
    model_path = root / 'animal2vec-nightingale.pt'
    metadata = {'metadata.json': ''}
    encoder = torch.jit.load(str(model_path), map_location='mps', _extra_files=metadata).eval()
    metadata = json.loads(metadata['metadata.json'])
    assert metadata['checkpoint_step'] == 10000
    reference = np.load(str(model_path) + '.verification.npz')
    checks = []
    with torch.no_grad():
        for waveform, expected in zip(reference['inputs'], reference['outputs'], strict=True):
            actual = encoder(torch.from_numpy(waveform).to('mps'))
            expected = torch.from_numpy(expected).to('mps')
            assert actual.device.type == 'mps' and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            checks.append(float((actual - expected).abs().max()))
    for invalid in [torch.zeros(1, 59999, device='mps'), torch.zeros(2, 60000, device='mps')]:
        try:
            encoder(invalid)
        except torch.jit.Error:
            pass
        else:
            raise AssertionError('Export accepted an unsupported input shape')
    verification = dict(python=sys.version, torch=torch.__version__, device='mps', cpu_fallback=False,
                        cases=['training audio 1', 'training audio 2', 'noise', 'silence'],
                        max_absolute_errors=checks, input_shape_checks='passed')
    benchmark.save_json(output / 'python314-verification.json', verification)
    print(json.dumps(verification), flush=True)
    base = root.parents[1] / 'scratch/tests'
    train_files = sorted((base / 'nightingale-train').glob('*.wav'))
    test_files = sorted((base / 'nightingale-test').glob('*.wav'))
    assert len(train_files) == 5 and len(test_files) == 2
    protocol = dict(json.loads((reference_root / 'protocol.json').read_text()),
                    encoder='Animal2vec nightingale, random initialization, 10000 updates',
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    window_seconds=metadata['window_seconds'], output_shape=metadata['output_shape'],
                    pretraining_split='Both test recording groups and paired channels excluded; validation segments excluded',
                    python_version=sys.version, torch_version=torch.__version__)
    if (output / 'protocol.json').exists():
        assert json.loads((output / 'protocol.json').read_text()) == protocol
    benchmark.save_json(output / 'protocol.json', protocol)
    train = [extract(wav, encoder, metadata, output) for wav in train_files]
    test = [extract(wav, encoder, metadata, output) for wav in test_files]
    assert sum(len(record['truth']) for record in test) == 7362
    del encoder
    gc.collect()
    torch.mps.empty_cache()
    benchmark.save_json(output / 'status.json', dict(stage='fitting_decoder', device='mps'))
    x = np.concatenate([r['features'][r['train_mask']] for r in train])
    y = np.concatenate([r['labels'][r['train_mask']] for r in train])
    fit_started = time.monotonic()
    weights, fitting = benchmark.fit_decoder(x, y)
    fit_seconds = time.monotonic() - fit_started
    np.savez(output / 'linear_decoder.npz', weights=weights.cpu().numpy(), classes=[0, 1])
    del x
    torch.mps.empty_cache()
    for record in train + test:
        record['probabilities'] = benchmark.probabilities(record['features'], weights)
    spec = importlib.util.spec_from_file_location('evaluation_reference', reference_root / 'evaluation_reference.py')
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    benchmark.save_json(output / 'status.json', dict(stage='validation_threshold_calibration'))
    threshold, calibration = benchmark.calibrate(train, metrics)
    benchmark.save_json(output / 'calibration.json', calibration)
    benchmark.save_json(output / 'validation.json', benchmark.evaluate(train, threshold, metrics, validation=True))
    fixed = benchmark.evaluate(test, 0.5, metrics)
    calibrated = benchmark.evaluate(test, threshold, metrics)
    benchmark.save_json(output / 'test-fixed-0.5.json', fixed)
    benchmark.save_json(output / 'test-calibrated.json', calibrated)
    encoder = torch.jit.load(str(model_path), map_location='mps').eval()
    segmenter = torch.jit.freeze(torch.jit.script(Segmenter(encoder, weights).eval()))
    waveform = torch.from_numpy(reference['inputs'][0]).to('mps')
    with torch.no_grad():
        torch.testing.assert_close(segmenter(waveform), (encoder(waveform) @ weights).sigmoid())
    torch.jit.save(segmenter, str(output / 'nightingale-segmenter.pt'),
                   _extra_files={'metadata.json': json.dumps(dict(metadata, threshold=threshold,
                       output_shape=metadata['output_shape'][:2], output='vocal probability'))})
    reloaded = torch.jit.load(str(output / 'nightingale-segmenter.pt'), map_location='mps').eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(waveform), segmenter(waveform))
    result = dict(encoder=metadata, device='mps', threshold=threshold, fitting=fitting,
                  train_examples=len(y), positive_train_examples=int(y.sum()), fitting_seconds=fit_seconds,
                  elapsed_seconds=time.monotonic() - started, fixed_metrics=fixed['metrics'],
                  calibrated_metrics=calibrated['metrics'],
                  validation_start_seconds={r['name']: r['validation_start'] for r in train},
                  extraction_seconds={r['name']: r['extraction_seconds'] for r in train + test})
    benchmark.save_json(output / 'summary.json', result)
    benchmark.save_json(output / 'status.json', dict(stage='complete'))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.reference_root.resolve())
