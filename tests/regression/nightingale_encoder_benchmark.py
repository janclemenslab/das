"""Remote frozen-encoder comparison on the existing nightingale train/test split."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import lightning as L
import numpy as np
import soundfile as sf
import torch

from das.models.model import DASModel
from das.models.pretrained import PretrainedAudioEncoder, PretrainedEncoderConfig

DEVICE = torch.device('mps')


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def labels_at(times, truth):
    ordered = np.asarray(sorted(truth), dtype=float)
    indexes = np.searchsorted(ordered[:, 0], times, side='right') - 1
    ends = np.maximum.accumulate(ordered[:, 1])
    return ((indexes >= 0) & (times < ends[indexes.clip(0)])).astype(np.int64)


def intervals(mask, hop, duration):
    changes = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1) * hop
    stops = np.minimum(np.flatnonzero(changes == -1) * hop, duration)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def clipped_truth(truth, start, stop):
    return [(max(on, start), min(off, stop)) for on, off in truth if off > start and on < stop]


def extract(wav, config, output, root):
    started = time.monotonic()
    annotation = json.loads(wav.with_suffix('.json').read_text())
    assert set(annotation['cluster']) == {'vocal'}
    truth = list(zip(annotation['onset'], annotation['offset'], strict=True))
    info = sf.info(wav)
    encoder = PretrainedAudioEncoder(config, sr=info.samplerate)
    # Reuse one loaded backbone across recordings without registering it in a decoder checkpoint.
    if hasattr(extract, 'backbone'):
        object.__setattr__(encoder, '_backbone', extract.backbone)
        encoder._backbone_device = DEVICE
    hop = config.hop_seconds / config.frames_per_window
    frames = int(np.ceil(info.frames * config.frames_per_window / encoder.window_hop_samples))
    cache = output / 'features-mps' / (wav.stem + '.npy')
    if not cache.exists():
        audio, _ = sf.read(wav, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        features = np.empty((frames, config.embedding_dim), dtype=np.float32)
        window_count = int(np.ceil(info.frames / encoder.window_hop_samples))
        for index, start in enumerate(range(0, info.frames, encoder.window_hop_samples)):
            block = encoder._embed_window(audio[start:start + encoder.window_samples], DEVICE)
            first = index * config.frames_per_window
            count = min(config.frames_per_window, frames - first)
            features[first:first + count] = block.reshape(-1, config.embedding_dim)[:count]
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == window_count:
                status = dict(encoder=config.type, stage='embedding', device=str(DEVICE), recording=wav.name,
                              windows=index + 1, total_windows=window_count,
                              mps_allocated_gb=torch.mps.current_allocated_memory() / 1e9,
                              elapsed_seconds=time.monotonic() - started)
                save_json(root / 'status.json', status)
                print(json.dumps(status), flush=True)
        assert np.isfinite(features).all()
        np.save(cache, features)
        extract.backbone = encoder._backbone
    features = np.load(cache, mmap_mode='r')
    assert features.shape == (frames, config.embedding_dim)
    window_starts = np.arange(frames) // config.frames_per_window * config.hop_seconds
    boundary = 0.9 * info.duration
    validation_start = np.ceil(boundary / config.hop_seconds) * config.hop_seconds
    centers = np.minimum((np.arange(frames) + 0.5) * hop, info.duration - 1e-6)
    return dict(name=wav.name, features=features, truth=truth, hop=hop, duration=info.duration,
                labels=labels_at(centers, truth),
                train_mask=window_starts + config.window_seconds <= boundary,
                validation_mask=window_starts >= validation_start,
                validation_start=float(validation_start), extraction_seconds=time.monotonic() - started)


def evaluate(records, threshold, metrics, validation=False):
    pairs, frame_totals, files = [], np.zeros(3, dtype=np.int64), []
    for record in records:
        active = record['probabilities'] >= threshold
        truth = record['truth']
        if validation:
            active &= record['validation_mask']
            truth = clipped_truth(truth, record['validation_start'], record['duration'])
        predicted = intervals(active, record['hop'], record['duration'])
        counts = metrics.frame_counts(predicted, truth, record['duration'])
        pairs.append((predicted, truth))
        frame_totals += counts
        scores = metrics.event_metrics([(predicted, truth)])
        metrics.add_frame_metrics(scores, counts)
        files.append(dict(file=record['name'], metrics=scores, predicted=predicted))
    scores = metrics.event_metrics(pairs)
    metrics.add_frame_metrics(scores, frame_totals)
    return dict(threshold=threshold, metrics=scores, files=files)


def calibrate(records, metrics):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        matched = predicted_count = truth_count = 0
        for record in records:
            predicted = intervals((record['probabilities'] >= threshold) & record['validation_mask'],
                                  record['hop'], record['duration'])
            truth = clipped_truth(record['truth'], record['validation_start'], record['duration'])
            matched += len(metrics.match_intervals(predicted, truth, iou_threshold=0.5))
            predicted_count += len(predicted)
            truth_count += len(truth)
        rows.append(dict(threshold=float(threshold), iou_f1=2 * matched / max(1, predicted_count + truth_count)))
    best = max(rows, key=lambda row: (row['iou_f1'], -abs(row['threshold'] - 0.5)))
    return best['threshold'], rows


def fit_decoder(x, y):
    """Same balanced L2 logistic objective as sklearn C=1, optimized on MPS."""
    features = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    target = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    scale = features.std(dim=0, correction=0)
    scale = torch.where(scale > 0, scale, 1)
    features = features / scale
    positives = target.sum()
    assert 0 < positives < len(y)
    balance = torch.where(target.bool(), len(y) / (2 * positives), len(y) / (2 * (len(y) - positives)))
    weight = torch.zeros(features.shape[1], device=DEVICE, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], max_iter=1000, tolerance_grad=1e-5,
                                 tolerance_change=1e-9, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(features @ weight, target, weight=balance)
        loss = loss + weight.square().sum() / (2 * len(y))
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = float(closure().detach())
    gradient = float(weight.grad.abs().max())
    assert np.isfinite(loss) and gradient < 5e-4, (loss, gradient)
    return (weight / scale).detach(), dict(iterations=optimizer.state[weight]['n_iter'],
                                         loss=loss, gradient_max=gradient)


@torch.no_grad()
def probabilities(features, weights):
    return (torch.as_tensor(np.asarray(features), dtype=torch.float32, device=DEVICE) @ weights).sigmoid().cpu().numpy()


def run(kind, root, train_dir, test_dir, metrics):
    started = time.monotonic()
    output = root / kind
    (output / 'features-mps').mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path.home() / '.cache/das/avex/avex_aves2.pt2') if kind == 'aves2' else None
    config = PretrainedEncoderConfig(type=kind, checkpoint=checkpoint, hop_seconds=1 if kind == 'aves2' else 5)
    train_files, test_files = sorted(train_dir.glob('*.wav')), sorted(test_dir.glob('*.wav'))
    assert len(train_files) == 5 and len(test_files) == 2
    train = [extract(path, config, output, root) for path in train_files]
    test = [extract(path, config, output, root) for path in test_files]
    assert sum(len(record['truth']) for record in test) == 7362
    if hasattr(extract, 'backbone'):
        del extract.backbone
    torch.mps.empty_cache()
    save_json(root / 'status.json', dict(encoder=kind, stage='fitting_decoder', device=str(DEVICE)))
    x = np.concatenate([record['features'][record['train_mask']] for record in train])
    y = np.concatenate([record['labels'][record['train_mask']] for record in train])
    fit_started = time.monotonic()
    weights, fitting = fit_decoder(x, y)
    fit_seconds = time.monotonic() - fit_started
    for record in train + test:
        record['probabilities'] = probabilities(record['features'], weights)
    np.savez(output / 'linear_decoder.npz', weights=weights.cpu().numpy(), classes=[0, 1])
    save_json(root / 'status.json', dict(encoder=kind, stage='validation_threshold_calibration'))
    threshold, calibration = calibrate(train, metrics)
    save_json(output / 'calibration.json', calibration)
    save_json(output / 'validation.json', evaluate(train, threshold, metrics, validation=True))
    save_json(root / 'status.json', dict(encoder=kind, stage='test_evaluation'))
    fixed = evaluate(test, 0.5, metrics)
    calibrated = evaluate(test, threshold, metrics)
    save_json(output / 'test-fixed-0.5.json', fixed)
    save_json(output / 'test-calibrated.json', calibrated)
    model = DASModel(num_classes=2, sr=44100, class_names=['noise', 'vocal'], class_types=['segment', 'segment'],
                     frontend={'type': 'raw'}, encoder=asdict(config), decoder={'type': 'linear'},
                     cross_entropy_weight=1, learning_rate=0.001, num_time_steps=2646000, chunk_stride=2646000)
    with torch.no_grad():
        model.decoder.decoder.weight.zero_()
        model.decoder.decoder.weight[1].copy_(weights.cpu())
    payload = {'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
               'pytorch-lightning_version': L.__version__}
    torch.save(payload, output / 'decoder.ckpt')
    loaded = DASModel.load_from_checkpoint(output / 'decoder.ckpt', map_location=DEVICE).to(DEVICE).eval()
    with torch.no_grad():
        actual = loaded.decoder(torch.as_tensor(x[:100], device=DEVICE)).softmax(-1)[:, 1].cpu().numpy()
    np.testing.assert_allclose(actual, probabilities(x[:100], weights), atol=1e-5, rtol=1e-4)
    import yaml
    (output / 'predict.yaml').write_text(yaml.safe_dump(dict(
        checkpoint=str(output / 'decoder.ckpt'), accelerator='mps', syllable_postprocessor='binary_mask',
        segment_threshold_low=threshold, segment_threshold_high=threshold, fill_gap_ms=0, min_syllable_ms=0)))
    result = dict(encoder=asdict(config), device=str(DEVICE), cpu_threads=4, train_files=[p.name for p in train_files],
                  test_files=[p.name for p in test_files], train_examples=len(y), positive_train_examples=int(y.sum()),
                  validation_start_seconds={r['name']: r['validation_start'] for r in train},
                  fitting_seconds=fit_seconds, fitting=fitting,
                  elapsed_seconds=time.monotonic() - started, threshold=threshold,
                  fixed_metrics=fixed['metrics'], calibrated_metrics=calibrated['metrics'],
                  extraction_seconds={r['name']: r['extraction_seconds'] for r in train + test})
    save_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def self_check():
    truth = [(1, 4), (2, 3), (5, 6)]
    assert labels_at(np.array([0, 1, 2.5, 3.5, 4, 5.5, 6]), truth).tolist() == [0, 1, 1, 1, 0, 1, 0]
    assert intervals([False, True, True, False, True], 0.5, 2.2) == [(0.5, 1.5), (2.0, 2.2)]
    assert clipped_truth(truth, 3.5, 5.5) == [(3.5, 4), (5, 5.5)]
    for window, hop, frames in [(10, 1, 1), (5, 5, 38)]:
        starts = np.arange(400) // frames * hop
        train = starts + window <= 47.2
        validation = starts >= np.ceil(47.2 / hop) * hop
        assert not (train & validation).any()
        assert (starts[train] + window <= 47.2).all()
    print('Label, interval, and context-separated validation checks passed.')


def gpu_check(root):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    assert torch.backends.mps.is_available(), 'MPS GPU is required'
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2000, 16)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] + rng.normal(size=len(x)) > 1).astype(np.int64)
    weights, fitting = fit_decoder(x, y)
    scaler = StandardScaler(with_mean=False)
    reference = LogisticRegression(C=1, fit_intercept=False, class_weight='balanced', tol=1e-8, max_iter=1000)
    reference.fit(scaler.fit_transform(x), y)
    error = float(np.max(np.abs(probabilities(x, weights) - reference.predict_proba(scaler.transform(x))[:, 1])))
    assert error < 0.001, error
    checks = dict(device=str(DEVICE), cpu_fallback=False, decoder_probability_max_error=error, fitting=fitting)
    wav = sorted((root.parents[1] / 'scratch/tests/nightingale-train').glob('*.wav'))[0]
    audio, sr = sf.read(wav, frames=10 * 44100, dtype='float32')
    for kind in ['birdcode', 'aves2']:
        checkpoint = str(Path.home() / '.cache/das/avex/avex_aves2.pt2') if kind == 'aves2' else None
        encoder = PretrainedAudioEncoder(PretrainedEncoderConfig(type=kind, checkpoint=checkpoint), sr=sr)
        times = []
        for _ in range(2):
            start = time.monotonic()
            features = encoder._embed_window(audio[:encoder.window_samples], DEVICE)
            torch.mps.synchronize()
            times.append(time.monotonic() - start)
        assert next(encoder._backbone.parameters()).device.type == 'mps'
        assert np.isfinite(features).all()
        checks[kind] = dict(shape=list(features.shape), seconds=times,
                            mps_allocated_gb=torch.mps.current_allocated_memory() / 1e9)
        print(json.dumps({kind: checks[kind]}), flush=True)
        del encoder
        torch.mps.empty_cache()
    save_json(root / 'gpu-check.json', checks)
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--gpu-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.gpu_check:
        torch.set_num_threads(4)
        gpu_check(args.root.resolve())
    else:
        assert torch.backends.mps.is_available(), 'MPS GPU is required; CPU fallback is disabled'
        root = args.root.resolve()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        spec = importlib.util.spec_from_file_location('evaluation_reference', root / 'evaluation_reference.py')
        metrics = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = metrics
        spec.loader.exec_module(metrics)
        base = root.parents[1] / 'scratch/tests'
        save_json(root / 'protocol.json', dict(
            started_at=datetime.now(timezone.utc).isoformat(), train_dir=str(base / 'nightingale-train'),
            test_dir=str(base / 'nightingale-test'), validation='last 10% per training recording; discard crossing contexts',
            decoder='MPS frozen embeddings; scaled without centering; balanced L2 logistic regression, C=1, no intercept, PyTorch LBFGS on MPS',
            calibration='Validation event IoU F1, thresholds 0.05 to 0.95 in steps of 0.05',
            test_metrics='event IoU >= 0.5; onset/offset collar 10 ms; frame metrics on 1 ms grid',
            test_split_limit='Chimichanga CH2 is in training and CH1 in test; Clipper is a separate recording',
            device=str(DEVICE), cpu_fallback=False, torch_version=torch.__version__, python_version=sys.version))
        results = {kind: run(kind, root, base / 'nightingale-train', base / 'nightingale-test', metrics)
                   for kind in ['birdcode', 'aves2']}
        save_json(root / 'summary.json', results)
        save_json(root / 'status.json', dict(stage='complete'))
