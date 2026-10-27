"""Export an AVEX encoder in Python 3.13 for inference in DAS on Python 3.14.

Use the same PyTorch version for export and inference. AVEX is only needed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from avex import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", default="esp_aves2_sl_beats_all")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.model_id, return_features_only=True, device="cpu").eval().requires_grad_(False)
    waveform = torch.randn(1, round(args.sample_rate * args.window_seconds))
    with torch.no_grad():
        expected = model(waveform)
        exported = torch.export.export(model, (waveform,))
        # MPS attention tensors can have different strides; reshape can copy when needed.
        for node in exported.graph.nodes:
            if node.target == torch.ops.aten.view.default:
                node.target = torch.ops.aten.reshape.default
        exported.graph_module.recompile()
        torch.testing.assert_close(exported.module()(waveform), expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"sample_rate": args.sample_rate, "window_seconds": args.window_seconds, "embedding_dim": expected.shape[-1]}
    torch.export.save(exported, args.output, extra_files={"das_embedding.json": json.dumps(metadata)})
    np.savez_compressed(str(args.output) + ".verification.npz", inputs=waveform.numpy(), outputs=expected.numpy())
    print(f"Saved {args.output}: {metadata}")


if __name__ == "__main__":
    main()
