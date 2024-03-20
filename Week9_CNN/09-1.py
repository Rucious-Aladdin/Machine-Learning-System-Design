#2018112571 김수성

from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

#hyperparameter config
tf.random.set_seed(42)
lr = 1e-3
epochs = 1

#09-1 - (1)
model = keras.models.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1),padding = "same",
                        kernel_size = (2, 2), filters = 4, activation="relu"),
    keras.layers.MaxPooling2D((2, 2), strides = 2),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_images, train_labels, batch_size=32, epochs=epochs,
                   validation_data=(test_images, test_labels))

print("test_정확도: %f%%" % (float(history.history["val_accuracy"][0]) * 100))
print("test_손실: %f" % history.history["val_loss"][0])

#09-1 - (2)
model = keras.models.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1),padding = "same",
                        kernel_size = (3, 3), filters = 64, activation="relu"),
    keras.layers.MaxPooling2D((2, 2), strides = 2),
    keras.layers.Conv2D(padding = "same", kernel_size = (3, 3), filters = 64, activation="relu"),
    keras.layers.Conv2D(padding = "same", kernel_size = (3, 3), filters = 32, activation="relu"),
    keras.layers.MaxPooling2D((2, 2), strides = 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_images, train_labels, batch_size=32, epochs=epochs,
                   validation_data=(test_images, test_labels))

print("test_정확도: %f%%" % (float(history.history["val_accuracy"][0]) * 100))
print("test_손실: %f" % history.history["val_loss"][0])