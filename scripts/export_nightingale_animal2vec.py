"""Export our completed nightingale checkpoint using its isolated upstream runtime.

The resulting TorchScript file needs only PyTorch, including on Python 3.14.
Input: one 2.5-second mono waveform at 24 kHz. Output: temporal embeddings.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import numpy as np
import torch


class Embeddings(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape[-1:])
        layers = self.model.extract_features(waveform, mask=False)['layer_results']
        return sum(layers[-12:]) / 12


class FixedWindow(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        torch._assert(waveform.dim() == 2, 'Expected [1, 60000] mono waveform')
        torch._assert(waveform.size(0) == 1 and waveform.size(1) == 60000,
                      'Expected one 2.5-second window at 24 kHz')
        return self.model(waveform)


def main(root):
    sys.path.insert(0, str(root / 'upstream'))
    from nn.data2vec2 import Data2VecMultiModel
    from omegaconf import OmegaConf
    from pretrain_nightingale_animal2vec import model_config

    assert torch.backends.mps.is_available(), 'MPS GPU is required'
    assert json.loads((root / 'status.json').read_text()) == dict(stage='complete', step=10000)
    torch.set_num_threads(4)
    torch.mps.profiler.start(wait_until_completed=True)
    checkpoint = root / 'checkpoint-last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['step'] == 10000
    config = model_config(root, 10000, 2)
    assert OmegaConf.to_container(config, enum_to_str=True) == state['config']
    model = Data2VecMultiModel.build_model(config.model)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.remove_pretraining_modules()
    model = Embeddings(model).eval().requires_grad_(False).to('mps')
    audio = np.load(sorted((root / 'audio-24k').glob('*.npy'))[0], mmap_mode='r')
    rng = np.random.default_rng(17)
    inputs = np.stack([audio[:60000], audio[240000:300000],
                       rng.normal(0, 0.1, 60000).astype(np.float32), np.zeros(60000, np.float32)])
    with torch.no_grad():
        expected = [model(torch.from_numpy(x[None]).to('mps')).cpu().numpy() for x in inputs]
        traced = torch.jit.trace(model, torch.from_numpy(inputs[:1]).to('mps'), check_trace=False)
        exported = torch.jit.freeze(torch.jit.script(FixedWindow(traced).eval()))
        # Tracing Python scalars can leave float64 constants that MPS cannot load.
        for node in exported.graph.nodes():
            if node.kind() == 'prim::Constant' and node.hasAttribute('value') and node.kindOf('value') == 't':
                value = node.t('value')
                if value.dtype == torch.float64:
                    value = value.float()
                    node.t_('value', value)
                    node.output().setType(torch._C.TensorType.create_from_tensor(value))
        for x, reference in zip(inputs, expected):
            actual = exported(torch.from_numpy(x[None]).to('mps'))
            torch.testing.assert_close(actual, torch.from_numpy(reference).to('mps'), atol=1e-4, rtol=1e-4)
            assert torch.isfinite(actual).all()
    output = root / 'animal2vec-nightingale.pt'
    metadata = dict(sample_rate=24000, window_seconds=2.5, input_shape=[1, 60000],
                    output_shape=list(expected[0].shape), embedding='mean of all 12 student layer results',
                    normalization='per-window layer_norm, eps=1e-5, included in model',
                    checkpoint_step=10000, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    torch_version=torch.__version__, device='mps', cpu_fallback=False)
    torch.jit.save(exported, str(output), _extra_files={'metadata.json': json.dumps(metadata)})
    reloaded = torch.jit.load(str(output), map_location='mps').eval()
    with torch.no_grad():
        for x, reference in zip(inputs, expected):
            actual = reloaded(torch.from_numpy(x[None]).to('mps'))
            torch.testing.assert_close(actual, torch.from_numpy(reference).to('mps'), atol=1e-4, rtol=1e-4)
    np.savez_compressed(str(output) + '.verification.npz', inputs=inputs[:, None], outputs=np.stack(expected))
    (root / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch.mps.profiler.stop()
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
