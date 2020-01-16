import tensorflow as tf


class Normalization2D(tf.keras.layers.Layer):
    """
    ### `Normalization2D`
    `kapre.utils.Normalization2D`
    A layer that normalises input data in ``axis`` axis.
    #### Parameters
    * input_shape: tuple of ints
    * str_axis: str
        - used ONLY IF ``int_axis`` is ``None``.
        - ``'batch'``, ``'data_sample'``, ``'channel'``, ``'freq'``, ``'time')``
        - Even though it is optional, actually it is recommended to use
        - ``str_axis`` over ``int_axis`` because it provides more meaningful
        - and image data format-robust interface.
    * int_axis: int
        - axis index that along which mean/std is computed.
        - `0` for per data sample, `-1` for per batch.
        - `1`, `2`, `3` for channel, row, col (if channels_first)
        - if `int_axis is None`, ``str_axis`` SHOULD BE set.
    #### Example
    A frequency-axis normalization after a spectrogram::
        ```python
        model.add(Spectrogram())
        model.add(Normalization2D(str_axis='freq'))
        ```
    """

    def __init__(self, str_axis=None, int_axis=None, image_data_format='default',
                 eps=1e-10, **kwargs):
        assert not (int_axis is None and str_axis is None), \
            'In Normalization2D, int_axis or str_axis should be specified.'

        assert image_data_format in ('channels_first', 'channels_last', 'default'), \
            'Incorrect image_data_format: {}'.format(image_data_format)

        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.str_axis = str_axis
        if self.str_axis is None: # use int_axis
            self.int_axis = int_axis
        else: # use str_axis
            # warning
            if int_axis is not None:
                print('int_axis={} passed but is ignored, str_axis is used instead.'.format(int_axis))
            # do the work
            assert str_axis in ('batch', 'data_sample', 'channel', 'freq', 'time'), \
                'Incorrect str_axis: {}'.format(str_axis)
            if str_axis == 'batch':
                int_axis = -1
            else:
                if self.image_data_format == 'channels_first':
                    int_axis = ['data_sample', 'channel', 'freq', 'time'].index(str_axis)
                else:
                    int_axis = ['data_sample', 'freq', 'time', 'channel'].index(str_axis)

        assert int_axis in (-1, 0, 1, 2, 3), 'invalid int_axis: ' + str(int_axis)
        self.axis = int_axis
        self.eps = eps
        super(Normalization2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.axis == -1:
            mean = tf.keras.backend.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            std = tf.keras.backend.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = tf.keras.backend.mean(x, axis=all_dims, keepdims=True)
            std = tf.keras.backend.std(x, axis=all_dims, keepdims=True)
        return (x - mean) / (std + self.eps)

    def get_config(self):
        config = {'int_axis': self.axis,
                  'str_axis': self.str_axis,
                  'image_data_format': self.image_data_format}
        base_config = super(Normalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))