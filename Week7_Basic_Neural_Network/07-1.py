#2018112571 김수성

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

class Loss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t
        return 0.5 * np.sum((self.y - self.t) ** 2)

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

class TwoLayerNet:
  def __init__(self):
    self.w = None
    self.grad = None
    self.weight_dict = {}
    self.weight_dict["W1"] = np.array([[0.2, 0.2, 0.3], [0.3, 0.1, 0.2]])
    self.weight_dict["b1"] = np.zeros(3)

    self.weight_dict["W2"] = np.array([[0.3, 0.2], [0.1, 0.4], [0.2, 0.3]])
    self.weight_dict["b2"] = np.zeros(2)

    self.layers = OrderedDict()
    self.layers["Affine1"] = Affine(self.weight_dict["W1"], self.weight_dict["b1"])
    self.layers["Sigmoid1"] = Sigmoid()
    self.layers["Affine2"] = Affine(self.weight_dict["W2"], self.weight_dict["b2"])
    self.layers["Sigmoid2"] = Sigmoid()
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

      return grads

if __name__ =="__main__":
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  Y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

  learning_rate = 1.0
  epoch = 1000

  model = TwoLayerNet()

  loss_list = []
  for i in range(epoch):
    for j in range(X.shape[0]):
      x = X[j].reshape(1, -1)
      y = Y[j].reshape(1, -1)
      grad = model.gradient(x, y)
      for key in ("W1", "b1", "W2", "b2"):
          model.weight_dict[key] -= learning_rate * grad[key]
    loss = model.loss(X, Y)
    loss_list.append(loss)
  i = 0
  for x in X:
    x = x.reshape(1, -1)
    y = model.predict(x)
    print("x1=%d, x2=%d, y1=%f, y2=%f"% (X[i][0], X[i][1], y[0][0], y[0][1]))
    i += 1

  plt.plot(range(epoch), loss_list)
  plt.show()