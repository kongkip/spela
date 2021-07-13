import tensorflow as tf

from .utils import backend


class Filterbank(tf.keras.layers.Layer):
    """
    ### `Filterbank`


    `spela.filterbank.Filterbank(n_fbs=128, trainable_fb, sr=None, init='mel', fmin=0., fmax=None,
                                 bins_per_octave=12, image_data_format='default', **kwargs)`
    #### Notes
        Input/output are 2D image format.
        E.g., if channel_first,
            - input_shape: ``(None, n_ch, n_freqs, n_time)``
            - output_shape: ``(None, n_ch, n_mels, n_time)``
    #### Parameters
    * n_fbs: int
       - Number of filterbanks
    * sr: int
        - sampling rate. It is used to initialize ``freq_to_mel``.
    * init: str
        - if ``'mel'``, init with mel center frequencies and stds.
    * fmin: float
        - min frequency of filterbanks.
        - If `init == 'log'`, fmin should be > 0. Use `None` if you got no idea.
    * fmax: float
        - max frequency of filterbanks.
        - If `init == 'log'`, fmax is ignored.
    * trainable: bool,
        - Whether the filterbanks are trainable or not.
    """

    def __init__(self, n_fbs, trainable, sr=16000, init="mel", fmin=0., fmax=None,
                 bins_per_octave=12, image_data_format="default", **kwargs):
        super(Filterbank, self).__init__(**kwargs)
        self.supports_masking = True
        self.n_fbs = n_fbs
        assert init in ("mel", "log", "linear", "uni_random")
        assert sr is not None
        if fmax is None:
            self.fmax = sr / 2.0
        else:
            self.fmax = fmax
        if init in ("mel", "log"):
            assert sr is not None

        self.fmin = fmin
        self.init = init
        self.bins_per_octave = bins_per_octave
        self.sr = sr
        self.trainable = trainable
        self.filterbank = None
        assert image_data_format in ("default", "channel_first", "channel_last")
        if image_data_format == "default":
            self.image_data_format = tf.keras.backend.image_data_format()
        else:
            self.image_data_format = image_data_format

    def build(self, input_shape):
        super(Filterbank, self).build(input_shape)
        self.built = True
        if self.image_data_format == "channel_first":
            self.n_ch = input_shape[1]
            self.n_freq = input_shape[2]
            self.n_time = input_shape[3]
        else:
            self.n_ch = input_shape[3]
            self.n_freq = input_shape[1]
            self.n_time = input_shape[2]

        if self.init == "mel":
            self.filterbank = tf.keras.backend.variable(backend.filterbank_mel(sr=self.sr,
                                                                               n_freq=self.n_freq,
                                                                               n_mels=self.n_fbs,
                                                                               fmin=self.fmin,
                                                                               fmax=self.fmax).transpose(),
                                                        dtype=tf.keras.backend.floatx())
        elif self.init == "log":
            self.filterbank = tf.keras.backend.variable(backend.filterbank_log(sr=self.sr,
                                                                               n_freq=self.n_freq,
                                                                               n_bins=self.n_bs,
                                                                               bins_per_octave=self.bins_per_octave,
                                                                               fmin=self.fmin).transpose(),
                                                        dtype=tf.keras.backend.floatx())
        if self.trainable:
            self.trainable_variables.append(self.filterbank)
        else:
            self.non_trainable_variables.append(self.filterbank)

    def compute_output_shape(self, input_shape):
        if self.image_data_format == "channel_first":
            return input_shape[0], self.n_ch, self.n_fbs, self.n_time
        else:
            return input_shape[0], self.n_fbs, self.n_time, self.n_ch

    def call(self, inputs, **kwargs):
        # reshape so that the last axis id freq axis
        if self.image_data_format == "channel_first":
            x = tf.keras.backend.permute_dimensions(inputs, (0, 1, 3, 1))
        else:
            x = tf.keras.backend.permute_dimensions(inputs, [0, 3, 2, 1])
        output = tf.keras.backend.dot(x, self.filterbank)
        # reshape back
        if self.image_data_format == "channel_first":
            return tf.keras.backend.permute_dimensions(output, [0, 1, 3, 2])
        else:
            return tf.keras.backend.permute_dimensions(output, [0, 3, 2, 1])

    def get_config(self):
        base_config = super(Filterbank, self).get_config()
        config = {"n_fbs": self.n_fbs,
                  "sr": self.sr,
                  "init": self.init,
                  "fmin": self.fmin,
                  "fmax": self.fmax,
                  "bins_per_octave": self.bins_per_octave,
                  "trainable": self.trainable}
        return dict(list(config.items()) + list(base_config.items()))
