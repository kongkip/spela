# SPELA - spectrogram layers

Rewrote [kapre](https://github.com/kongkip/kapre#installation) using tensorflow.keras \
credits go to Keunwoo Choi for writing kapre

My main goal for rewriting it with tensorflow.keras is to use it with TensorFlow Lite \
since Keunwoo Choi used core keras and I had problems converting the model to \
tensorflow lite.

Implementing audio features inside the keras layers allows the preprocessing \
computations to be done on the GPU as highlighted in their [paper](https://arxiv.org/abs/1706.05781)

Checkout [this]() Speaker Recognition project to see the usage of Spela.

# Installation

The package uses tensorflow but is not listed as requirement, please install it.

```bash
pip install spela
```

or

```bash
git clone https://github.com/kongkip/spela.git
cd spela
python setup.py install
```

# Usage

## spectrogram

```python
import tensorflow as tf
from spela.spectrogram import Spectrogram

# a one channel audio with 16000 sample rate
input_shape = (1, 16000)

x = get_data()
y = get_data()


model = tf.keras.Sequential()
model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(input_shape),
                      return_decibel_spectrogram=True, power_spectrogram=2.0,
                      trainable_kernel=False, name='static_stft'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.categorical_crossentropy
              , metrics=["acc"])

print(model.summary())

model.fit(x,y)
```

## Mel Spectrogram

```python
import tensorflow as tf
from spela.melspectrogram import Melspectrogram

# a one channel audio with 16000 sample rate
input_shape = (1, 16000)

x = get_data()
y = get_data()

model = tf.keras.Sequential()
model.add(Melspectrogram(sr=SR, n_mels=128,
          n_dft=512, n_hop=256, input_shape=input_shape,
          return_decibel_melgram=True,
          trainable_kernel=False, name='melgram'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.categorical_crossentropy
              , metrics=["acc"])

print(model.summary())

model.fit(x,y)
```
