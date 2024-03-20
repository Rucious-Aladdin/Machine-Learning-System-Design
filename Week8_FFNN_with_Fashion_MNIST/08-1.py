#2018112571 김수성

from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25)
print(train_images.shape, train_labels.shape)
print(val_images.shape, val_labels.shape)


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_images, train_labels, batch_size=64, epochs=10,
                    validation_data=(val_images, val_labels))

train_loss = history.history['loss']
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

x = np.arange(len(train_loss))
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.grid(True)
plt.plot(x, train_loss, color="blue", label="train_loss")
plt.plot(x, val_loss, color="red", label="val_loss")

plt.subplot(122)
plt.plot(x, train_acc, color="blue", label="train_acc")
plt.plot(x, val_acc, color="red", label="val_acc")
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()
plt.clf()

#08-2-(2)

labels_dict = {0: "T-shirt/top",
               1: "Trouser",
               2: "Pullover",
               3: "Dress",
               4: "Coat",
               5: "Sandal",
               6: "Shirt",
               7: "Sneaker",
               8: "Bag",
               9: "Ankle boot"}

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(labels_dict[test_labels[i]])
plt.show()