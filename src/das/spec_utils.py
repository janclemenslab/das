import logging
import tensorflow.keras as keras
import tensorflow as tf
try:
    import tensorflow_probability as tfp
except ImportError:
    tfp = None
    logging.debug('No tensorflow-probability.')
    pass


class MelSpec(keras.layers.Layer):
    # copied from https://keras.io/examples/audio/melgan_spectrogram_inversion

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

        if frame_step is None:
            frame_step = self.frame_length // 4
        self.frame_step = frame_step

        if fft_length is None:
            fft_length = self.frame_length * 2
        self.fft_length = fft_length

        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min

        if freq_max is None:
            freq_max = self.sampling_rate // 2
        self.freq_max = freq_max

        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_channels,
            num_spectrogram_bins=self.fft_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, audio, training=True):
        # We will only perform the transformation during training.
        # if training:
        # Taking the Short Time Fourier Transform. Ensure that the audio is padded.
        # In the paper, the STFT output is padded using the 'REFLECT' strategy.
        stft = tf.signal.stft(
            tf.squeeze(audio, -1),
            self.frame_length,
            self.frame_step,
            self.fft_length,
            pad_end=True,
        )

        # Taking the magnitude of the STFT output
        magnitude = tf.abs(stft)

        # Multiplying the Mel-filterbank with the magnitude and scaling it using the db scale
        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)

        # # baseline subtract
        if tfp is not None:
            baseline = tfp.stats.percentile(mel, 50, axis=1, keepdims=True)
            mel = tf.math.log(mel / (baseline + 0.000001))
            mel = tf.clip_by_value(mel,
                                   clip_value_min=0,
                                   clip_value_max=100_000)
            mel /= tf.reduce_max(mel) * 255
        return mel

    def get_config(self):
        config = super(MelSpec, self).get_config()
        config.update({
            "frame_length": self.frame_length,
            "frame_step": self.frame_step,
            "fft_length": self.fft_length,
            "sampling_rate": self.sampling_rate,
            "num_mel_channels": self.num_mel_channels,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
        })
        return config
