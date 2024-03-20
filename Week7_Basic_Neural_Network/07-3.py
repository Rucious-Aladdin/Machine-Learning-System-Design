#2018112571 김수성 07-3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

path = "/content/drive/MyDrive/기계학습시스템설계/hw07/nonlinear.csv"
df = pd.read_csv(path)

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="tanh"),
    keras.layers.Dense(16, activation="tanh"),
    keras.layers.Dense(8, activation="tanh"),
    keras.layers.Dense(4, activation="tanh"),
    keras.layers.Dense(1, activation="tanh")
])

lr = 0.1
epoch = 100
optimizer = keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer=optimizer, loss="mse")

print(df.head())
X = df["x"].to_numpy().reshape(-1, 1)
Y = df["y"].to_numpy().reshape(-1, 1)

model.fit(X, Y, epochs=epoch)

domain = np.linspace(0, 1, 100).reshape(-1, 1)
y_hat = model.predict(domain)
plt.scatter(df["x"], df["y"])
plt.scatter(domain, y_hat, color="r")
plt.show()