#2018112571 김수성

from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train, y_train, x_val, y_val = x_train[:10000] / 255, y_train[:10000], x_val[:2000] / 255, y_val[:2000]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

x_train_preprocess = np.zeros((10000, 75, 75, 3))
x_val_preprocess = np.zeros((2000, 75, 75, 3))

for i, img in enumerate(x_train):
  img_resize = resize(img, (75, 75), anti_aliasing=True)
  x_train_preprocess[i] = np.dstack([img_resize, img_resize, img_resize])

for i, img in enumerate(x_val):
  img_resize = resize(img, (75, 75), anti_aliasing=True)
  x_val_preprocess[i] = np.dstack([img_resize, img_resize, img_resize])

print(x_train_preprocess.shape)
print(x_val_preprocess.shape)

#hyperparameter config
lr = 0.0005
epochs = 10

#model load
pre_trained_model = InceptionV3(input_shape=(75, 75, 3),
                                include_top=False,
                                weights="imagenet")
for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer("mixed7")
last_output = last_layer.output

x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dense(10, activation="softmax")(x)

model = keras.Model(pre_trained_model.input, x)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

train_loss = history.history['loss']
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

x = np.arange(len(train_loss))
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.grid(True)
plt.plot(x, train_loss, color="red", label="train_loss")
plt.plot(x, val_loss, color="blue", label="val_loss")
plt.legend()

plt.subplot(122)
plt.plot(x, train_acc, color="red", label="train_acc")
plt.plot(x, val_acc, color="blue", label="val_acc")
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()