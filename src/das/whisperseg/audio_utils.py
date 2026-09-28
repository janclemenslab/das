from transformers import WhisperFeatureExtractor
from transformers.audio_utils import mel_filter_bank


def get_n_fft_given_sr(sr):
    if sr <= 32000:
        n_fft = 512
    elif sr <= 80000:
        n_fft = 1024
    elif sr <= 150000:
        n_fft = 2048
    elif sr <= 300000:
        n_fft = 4096
    else:
        n_fft = 8192
    return n_fft


class WhisperSegFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, sr, spec_time_step, min_frequency=None, max_frequency=None, chunk_length=30):
        hop_length = int(spec_time_step * sr)
        n_fft = get_n_fft_given_sr(sr)

        if min_frequency is None:
            min_frequency = 0
        if max_frequency is None:
            max_frequency = sr // 2

        super().__init__(
            feature_size=80,
            sampling_rate=sr,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=0.0,
            return_attention_mask=False,
        )

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=80,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            sampling_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
