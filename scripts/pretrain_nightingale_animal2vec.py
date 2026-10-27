"""Train upstream Animal2vec from scratch using the audited nightingale manifest.

Run inside the isolated upstream Python 3.9 environment, not the DAS runtime.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_gpu_step(gradient, error_log):
    if error_log is not None and 'command buffer exited with error status' in error_log.read():
        raise RuntimeError('Metal command failed; refusing to update model weights')
    # Healthy updates were below 80; failed Metal commands produced norms above 1e7.
    if not math.isfinite(gradient) or gradient > 1000:
        raise RuntimeError(f'Abnormal gradient norm {gradient}; refusing to update model weights')


def prepare(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = [row for row in manifest['files'] if row['use'] == 'pretrain']
    held_out = set(manifest['held_out_recording_groups'])
    assert len({row['sha256'] for row in rows}) == len(rows)
    (root / 'audio-24k').mkdir(exist_ok=True)
    for row in rows:
        assert row['recording_group'] not in held_out
        assert row['path'].lower().endswith(('ch1.wav', 'ch2.wav'))
        source = root / 'raw' / row['path']
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == row['sha256'], source
        target = root / 'audio-24k' / (source.stem + '.npy')
        count = math.floor(row['train_stop_seconds'] * row['sample_rate'])
        with sf.SoundFile(source) as handle:
            assert handle.samplerate == 44100 and handle.channels == 1
            audio = handle.read(count, dtype='float32')
        # Crop BEFORE resampling so validation samples cannot enter the filter.
        audio = resample_poly(audio, 80, 147).astype(np.float32)
        assert np.isfinite(audio).all()
        np.save(target, audio)
        print(json.dumps(dict(prepared=source.name, samples=len(audio))), flush=True)
    save_json(root / 'prepared.json', dict(files=len(rows), sample_rate=24000,
              training_hours=sum(row['train_stop_seconds'] for row in rows) / 3600))


def model_config(root, steps, clones):
    from omegaconf import OmegaConf
    from nn.data2vec2 import Data2VecMultiConfig
    from nn.criterions import ExpandedModelCriterionConfig

    full = OmegaConf.load(root / 'upstream/configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml')
    del full.model['_name']
    del full.criterion['_name']
    full.model = OmegaConf.merge(OmegaConf.structured(Data2VecMultiConfig()), full.model)
    full.criterion = OmegaConf.merge(OmegaConf.structured(ExpandedModelCriterionConfig()), full.criterion)
    full.common.seed = 7
    full.optimization.max_update = steps
    full.model.clone_batch = clones
    full.model.ema_decay = 0.999
    full.model.ema_end_decay = 0.9999
    full.model.ema_anneal_end_step = steps
    full.model.log_norms = False
    return OmegaConf.create({'model': OmegaConf.to_container(full.model, resolve=True)})


def train(args):
    from omegaconf import OmegaConf
    from nn.data2vec2 import Data2VecMultiModel

    root = args.root
    torch.set_num_threads(4)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    config = model_config(root, args.steps, args.clones)
    model = Data2VecMultiModel.build_model(config.model)
    print(json.dumps(dict(stage='model_built', parameters=sum(p.numel() for p in model.parameters()),
                          initialization='random', device=args.device)), flush=True)
    if args.build_only:
        save_json(root / 'model-config.json', OmegaConf.to_container(config, enum_to_str=True))
        return
    assert (root / 'prepared.json').exists(), 'Run --prepare first'
    assert args.device == 'mps' and torch.backends.mps.is_available(), 'MPS GPU is required'
    error_log = args.stderr_log.open() if args.stderr_log else None
    if error_log is not None:
        error_log.seek(0, os.SEEK_END)
    # Flush GPU work per operation after native Metal command-buffer failures.
    torch.mps.profiler.start(wait_until_completed=True)
    model.to(args.device).train()
    groups = [dict(params=[], weight_decay=0.01), dict(params=[], weight_decay=0.0)]
    for name, parameter in model.named_parameters():
        no_decay = parameter.ndim == 1 or name.endswith('.bias') or 'alibi_scale' in name or 'p_swish' in name
        groups[int(no_decay)]['params'].append(parameter)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    checkpoint = root / ('smoke-last.pt' if args.smoke else 'checkpoint-last.pt')
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['config'] == OmegaConf.to_container(config, enum_to_str=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        model.num_updates = start
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if args.device == 'mps':
            torch.mps.set_rng_state(state['mps_rng'])
    audio = [np.load(path, mmap_mode='r') for path in sorted((root / 'audio-24k').glob('*.npy'))]
    samples = round(args.seconds * 24000)
    weights = np.asarray([len(x) - samples + 1 for x in audio], dtype=float)
    assert (weights > 0).all()
    weights /= weights.sum()
    run_config = dict(config=OmegaConf.to_container(config, enum_to_str=True),
                      crop_seconds=args.seconds, batch_size=args.batch_size, gradient_accumulation=args.accumulate,
                      steps=args.steps, learning_rate=args.lr, device=args.device,
                      synchronous_gpu_operations=True,
                      torch_version=torch.__version__, initialization='random; no pretrained checkpoint')
    save_json(root / ('smoke-config.json' if args.smoke else 'training-config.json'), run_config)
    log_path = root / ('smoke-metrics.jsonl' if args.smoke else 'metrics.jsonl')
    beginning = time.monotonic()
    final_step = min(start + 2, args.steps) if args.smoke else args.steps
    for step in range(start, final_step):
        optimizer.zero_grad(set_to_none=True)
        warmup = min(1000, max(1, args.steps // 10))
        rate = min((step + 1) / warmup, 1.0) * 0.5 * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, args.steps - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * rate
        loss_value = 0.0
        for _ in range(args.accumulate):
            batch = []
            for index in np.random.choice(len(audio), args.batch_size, p=weights):
                offset = np.random.randint(0, len(audio[index]) - samples + 1)
                batch.append(np.array(audio[index][offset:offset + samples]))
            source = torch.from_numpy(np.stack(batch)).to(args.device)
            source = torch.nn.functional.layer_norm(source, source.shape[-1:])
            result = model(source=source, mask=True)
            loss = sum(value.sum() for value in result['losses'].values()) / result['sample_size']
            assert torch.isfinite(loss), 'Non-finite loss'
            (loss / args.accumulate).backward()
            loss_value += float(loss.detach()) / args.accumulate
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        check_gpu_step(float(gradient), error_log)
        optimizer.step()
        model.set_num_updates(step + 1)
        torch.mps.synchronize()
        check_gpu_step(float(gradient), error_log)
        row = dict(step=step + 1, loss=loss_value, gradient_norm=float(gradient), learning_rate=args.lr * rate,
                   target_var=float(result['target_var']), pred_var=float(result['pred_var']),
                   elapsed_seconds=time.monotonic() - beginning)
        if args.device == 'mps':
            row['mps_allocated_gb'] = torch.mps.current_allocated_memory() / 1e9
            row['mps_driver_gb'] = torch.mps.driver_allocated_memory() / 1e9
        with log_path.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if (step + 1) % 10 == 0 or args.smoke or step == start:
            save_json(root / ('smoke-status.json' if args.smoke else 'status.json'), row)
            print(json.dumps(row), flush=True)
        if (step + 1) % 10 == 0 or step == start or step + 1 == final_step:
            state = dict(step=step + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         config=run_config['config'], torch_rng=torch.get_rng_state(),
                         numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                         mps_rng=torch.mps.get_rng_state() if args.device == 'mps' else None)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(checkpoint)
    if not args.smoke:
        save_json(root / 'status.json', dict(stage='complete', step=final_step))
    torch.mps.profiler.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two updates, with separate checkpoint and logs; rerun to check resume.')
    parser.add_argument('--stderr-log', type=Path, help='Capture stderr here when launching, to detect native Metal failures.')
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--clones', type=int, default=2)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    args.root = args.root.resolve()
    sys.path.insert(0, str(args.root / 'upstream'))
    if args.prepare:
        prepare(args.root)
    else:
        train(args)
