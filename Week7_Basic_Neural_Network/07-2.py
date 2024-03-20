#2018112571 김수성

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

class Loss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t.reshape(-1, 1)
        return 0.5 * np.sum((self.y - self.t) ** 2) / self.t.shape[0]

    def backward(self, dout=1):
        dx = (self.y - self.t)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out # 상류계층 미분 * y * (1-y)
        return dx

class Affine:
  def __init__(self, w, b):
    self.w = w
    self.b = b
    self.x = None

    self.dW = None
    self.db = None

  def forward(self, x):
    self.x = x
    return x.dot(self.w) + self.b

  def backward(self, dout):
    dx = np.dot(dout, self.w.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis = 0)

    return dx

class ThreeLayerNet:
  def __init__(self):
    self.w = None
    self.grad = None
    self.weight_dict = {}
    self.weight_dict["W1"] = np.random.randn(1, 6)
    self.weight_dict["b1"] = np.zeros(6)

    self.weight_dict["W2"] = np.random.randn(6, 4)
    self.weight_dict["b2"] = np.zeros(4)

    self.weight_dict["W3"] = np.random.randn(4, 1)
    self.weight_dict["b3"] = np.zeros(1)

    self.layers = OrderedDict()
    self.layers["Affine1"] = Affine(self.weight_dict["W1"], self.weight_dict["b1"])
    self.layers["Sigmoid1"] = Sigmoid()
    self.layers["Affine2"] = Affine(self.weight_dict["W2"], self.weight_dict["b2"])
    self.layers["Sigmoid2"] = Sigmoid()
    self.layers["Affine3"] = Affine(self.weight_dict["W3"], self.weight_dict["b3"])
    self.layers["Sigmoid3"] = Sigmoid()
    self.lastLayer = Loss()

  def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

  def loss(self, x, t):
      y = self.predict(x)
      return self.lastLayer.forward(y, t)

  def gradient(self, x, t):
      self.loss(x, t)
      dout = 1
      dout = self.lastLayer.backward(dout)
      layers = list(self.layers.values())
      layers.reverse()
      for i, layer in enumerate(layers):
          dout = layer.backward(dout)
      grads = {}
      grads["W1"] = self.layers["Affine1"].dW
      grads["b1"] = self.layers["Affine1"].db
      grads["W2"] = self.layers["Affine2"].dW
      grads["b2"] = self.layers["Affine2"].db
      grads["W3"] = self.layers["Affine3"].dW
      grads["b3"] = self.layers["Affine3"].db

      return grads

if __name__ =="__main__":
  path = "/content/drive/MyDrive/기계학습시스템설계/hw07/nonlinear.csv"
  df = pd.read_csv(path)
  X = df["x"].to_numpy().reshape(-1 ,1)
  Y = df["y"].to_numpy().reshape(-1, 1)
  learning_rate = 1.0
  epoch = 100
  model = ThreeLayerNet()

  loss_list = []
  for i in range(epoch):
    for j in range(X.shape[0]):
      x = X[j].reshape(1, 1)
      y = Y[j].reshape(1, 1)
      grad = model.gradient(x, y)
      for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
          model.weight_dict[key] -= learning_rate * grad[key]
    loss = model.loss(X, Y)
    loss_list.append(loss)

  plt.scatter(X, Y)
  x = np.arange(0, 1, 0.01).reshape(-1, 1)
  y = model.predict(x)
  plt.plot(x, y, linewidth=6, color="red")
  plt.show()