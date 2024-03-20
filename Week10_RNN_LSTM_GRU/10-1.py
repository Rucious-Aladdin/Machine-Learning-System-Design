#2018112571 김수성

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

x = np.arange(0, 10.2, 0.2)
print(x)

#data 생성
rand_unif = np.random.uniform(low=0.0, high=1.0, size=100)
phase_shift = rand_unif * 2 * np.pi

data = np.sin(x[:-1] - phase_shift[0])
label = np.sin(x[-1] - phase_shift[0])
for i, shift in enumerate(phase_shift):
  if (i != 0):
    temp_data  = np.sin(x[:-1] - shift)
    temp_label = np.sin(x[-1]- shift)
    data = np.row_stack((data, temp_data))  
    label = np.row_stack((label, temp_label))

subplot_index = 151
plt.figure(figsize = (25, 5))
for i in range(5):
  plt.subplot(subplot_index)
  plt.scatter(x[:-1], data[i])
  plt.scatter(x[-1], label[i])
  subplot_index += 1
plt.show()
print(data.shape)
print(label.shape)

#train test 데이터 분리

x_train, x_test = data[:80], data[80:]
t_train, t_test = label[:80], label[80:]
print(x_train.shape, x_test.shape)
print(t_train.shape, t_test.shape)

def plot(history, label, y_hat):
  plt.clf()
  plt.figure(figsize = (10, 10))
  plt.subplot(221)
  plt.plot(history.history["loss"])
  plt.ylabel("loss")
  plt.xlabel("epoch")

  plt.subplot(222)
  plt.scatter(t_train, y_hat[:80], label="train")
  plt.scatter(t_test, y_hat[80:], label="test")
  plt.xlabel("y")
  plt.ylabel("y_hat")
  plt.legend()

  plt.subplot(223)
  plt.plot(label[:80], color="black", label="y")
  plt.plot(y_hat[:80], color="red", label="y_hat")
  plt.text(0, 0.75, "MSE: %f" % mean_squared_error(label[:80], y_hat[:80]))
  
  plt.subplot(224)
  plt.plot(label[80:], color="black", label="y")
  plt.plot(y_hat[80:], color="red", label="y_hat")
  plt.text(0, 0.75, "MSE: %f" % mean_squared_error(label[80:], y_hat[80:]))

  plt.tight_layout()
  plt.show()

#RNN
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units =  10, return_sequences=False,
                              input_shape = [50, 1]),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()
print(x_train.shape, t_train.shape)

history = model.fit(x_train, t_train, epochs=50)
y_hat = model.predict(data)
plot(history, label, y_hat)

#LSTM
model_LSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(units =  10, return_sequences=False,
                              input_shape = [50, 1]),
    tf.keras.layers.Dense(1)
])

model_LSTM.compile(optimizer="adam", loss="mse")
model_LSTM.summary()
print(x_train.shape, t_train.shape)
history = model_LSTM.fit(x_train, t_train, epochs=50)
y_hat = model_LSTM.predict(data)
plot(history, label, y_hat)

#GRU
model_GRU = tf.keras.Sequential([
    tf.keras.layers.GRU(units =  10, return_sequences=False,
                              input_shape = [50, 1]),
    tf.keras.layers.Dense(1)
])

model_GRU.compile(optimizer="adam", loss="mse")
model_GRU.summary()
print(x_train.shape, t_train.shape)
history = model_GRU.fit(x_train, t_train, epochs=50)
y_hat = model_GRU.predict(data)
plot(history, label, y_hat)

