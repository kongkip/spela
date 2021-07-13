import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from spela.melspectrogram import Melspectrogram

SR = 16000
wav = librosa.load("examples/data/62.wav", sr=16000)[0]
print(wav.shape)
src = np.random.random((1, SR * 3))
new = wav[np.newaxis, np.newaxis, :]
print(new.shape)
print(src.shape)

model = tf.keras.Sequential()
model.add(Melspectrogram(sr=SR, n_mels=128,
                         n_dft=512, n_hop=256, input_shape=(new.shape[1], new.shape[2]),
                         return_decibel_melgram=True,
                         trainable_kernel=False, name='melgram'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.categorical_crossentropy
              , metrics=["acc"])

print(model.summary())

pred = model.predict(x=new)

result = pred[0, :, :, 0]
librosa.display.specshow(result)
plt.show()
