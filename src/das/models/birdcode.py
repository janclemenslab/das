"""BirdCODE's trained ConvNeXt features before its species-prototype head.

Preprocessing and spatial upsampling follow the upstream MIT-licensed runtime:
https://github.com/earthspecies/sound-event-detection/blob/main/src/sound_event_detection/models/audioprotopnet_frame_detector.py

MIT License — Copyright (c) 2026 Earth Species Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json

import torch
from torch import nn
from torch.nn import functional as F


class BirdCODEEmbeddings(nn.Module):
    """Five seconds of 32 kHz audio -> [batch, 38, 1024] frequency-pooled features."""

    def __init__(self, model_id: str):
        super().__init__()
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from torchaudio import transforms as T
        from transformers import ConvNextConfig, ConvNextModel

        config = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
        if config["encoder"]["type"] != "audioprotopnet_frame" or (config["sample_rate"], config["target_duration"]) != (
            32000,
            5,
        ):
            raise ValueError("Expected the five-second AudioProtoPNet BirdCODE checkpoint.")
        weights = torch.load(
            hf_hub_download(model_id, "checkpoint/best_model.pt"), map_location="cpu", weights_only=True, mmap=True
        )
        self.backbone = ConvNextModel(ConvNextConfig.from_pretrained("facebook/convnext-base-224-22k", num_channels=1))
        prefix = "encoder.backbone."
        self.backbone.load_state_dict(
            {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
        )
        self.mel_transform = nn.Sequential(
            T.Spectrogram(n_fft=2048, hop_length=256, power=2.0),
            T.MelScale(n_mels=256, sample_rate=32000, n_stft=1025),
            T.AmplitudeToDB(stype="power", top_db=80),
        )
        self.mel_transform[0].window.copy_(weights["encoder.mel_transform.spectrogram.window"])
        self.mel_transform[1].fb.copy_(weights["encoder.mel_transform.mel_scale.fb"])

    def forward(self, waveform):
        mel = (self.mel_transform(waveform) + 13.369) / 13.162
        features = self.backbone(mel.unsqueeze(1)).last_hidden_state
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        return features.mean(dim=2).transpose(1, 2)
