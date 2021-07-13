import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from librosa.display import waveplot, specshow

from spela.spectrogram import Spectrogram

SR = 8000
wav = librosa.load("examples/data/62.wav", sr=8000)[0]
print(wav.shape)
src = np.random.random((1, SR * 3))
new = wav[np.newaxis, np.newaxis, :]
print(new.shape)
print(src.shape)
# waveplot(wav)
# plt.show()

#
#
#

height = new.shape[1]
width = new.shape[2]

model = tf.keras.Sequential()
# return_decibel_spectrogram means it returns decibel of the power spectrogram
model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(height, width),
                      return_decibel_spectrogram=True, power_spectrogram=2.0,
                      trainable_kernel=False, name='static_stft'))
# model.add(Normalization2D(str_axis='freq'))


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss="categorical_crossentropy"
              , metrics=[tf.keras.metrics.categorical_accuracy])

print(model.summary())

pred = model.predict(x=new)

if tf.keras.backend.image_data_format() == "channel_first":
    result = pred[0, 0]
else:
    result = pred[0, :, :, 0]

specshow(result, y_axis='linear', sr=SR)
plt.show()

# export the model as TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
TFLite_model = converter.convert()
open("tflite_model.tflite", "wb").write(TFLite_model)

interpreter = tf.lite.Interpreter(model_content=TFLite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details: {}  Output details {}".format(input_details,
                                                    output_details))

input_shape = input_details[0]["shape"]
input_data = new
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[0]["index"])

# print("TensorFlow Lite results", tflite_results)
tflite_results = tflite_results[0, :, :, 0]
librosa.display.specshow(tflite_results,
                         y_axis='linear', sr=SR)
plt.show()
