"""Backward-compatible spectrogram layers."""

import keras


class MelSpec(keras.layers.Layer):
    """Torch-backed replacement for the legacy TensorFlow mel-spectrogram layer."""

    def __init__(
        self,
        sampling_rate,
        frame_length=512,
        frame_step=None,
        fft_length=None,
        num_mel_channels=128,
        freq_min=0,
        freq_max=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_length // 4 if frame_step is None else frame_step
        self.fft_length = frame_length * 2 if fft_length is None else fft_length
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = sampling_rate // 2 if freq_max is None else freq_max
        self._mel = keras.layers.MelSpectrogram(
            fft_length=self.fft_length,
            sequence_stride=self.frame_step,
            sequence_length=self.frame_length,
            sampling_rate=self.sampling_rate,
            num_mel_bins=self.num_mel_channels,
            min_freq=self.freq_min,
            max_freq=self.freq_max,
            power_to_db=False,
        )

    def call(self, audio, training=None):
        del training
        mel = self._mel(keras.ops.squeeze(audio, axis=-1))
        return keras.ops.transpose(mel, (0, 2, 1))

    def get_config(self):
        return {
            **super().get_config(),
            "sampling_rate": self.sampling_rate,
            "frame_length": self.frame_length,
            "frame_step": self.frame_step,
            "fft_length": self.fft_length,
            "num_mel_channels": self.num_mel_channels,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
        }
