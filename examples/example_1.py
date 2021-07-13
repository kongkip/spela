import tensorflow as tf
import numpy as np
import librosa
from spela.spectrogram import Spectrogram

# import mute_tf_warning as mw

SR = 16000
wav = librosa.load("examples/data/62.wav", sr=SR)[0]
wav.shape
new_wav = wav[np.newaxis, np.newaxis, :]
new_wav.shape

model = tf.keras.Sequential()
model.add(Spectrogram(n_dft=512, n_hop=256,
                    input_shape=(1,16000),
                      return_decibel_spectrogram=True, power_spectrogram=2.0,
                      trainable_kernel=False, name='static_stft'))
model.add(tf.keras.layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss=tf.keras.losses.categorical_crossentropy
              , metrics=["acc"])

print(model.summary())

model.predict(new_wav)
