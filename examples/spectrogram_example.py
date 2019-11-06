import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from spela.spectrogram import Spectrogram
import mute_tf_warnings as mw

mw.tf_mute_warning()

SR = 16000
wav = librosa.load("examples/data/62.wav",sr=16000)[0]
print(wav.shape)
src = np.random.random((1, SR * 3))
new = wav[np.newaxis, np.newaxis, :]
print(new.shape)
print(src.shape)
# librosa.display.waveplot(wav)
# plt.show()

#
#
#
model = tf.keras.Sequential()
model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(new.shape[1],new.shape[2]),
                      return_decibel_spectrogram=True, power_spectrogram=2.0,
                      trainable_kernel=False, name='static_stft'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.categorical_crossentropy
              , metrics=["acc"])

print(model.summary())

pred = model.predict(x=new)

if tf.keras.backend.image_data_format() == "channel_first":
    result = pred[0, 0]
else:
    result = pred[0, :, :, 0]


# result = librosa.power_to_db(result)
librosa.display.specshow(result,
                 y_axis='linear', sr=SR)
plt.show()

#export the model as tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("tflitemodel.tflite", "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details: {}  Output details {}".format(input_details, output_details))

input_shape = input_details[0]["shape"]
input_data = new
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[0]["index"])

# print("Tensorflow Lite results", tflite_results)
tflite_results = tflite_results[0, :, :, 0]
librosa.display.specshow(tflite_results,
                 y_axis='linear', sr=SR)
plt.show()

