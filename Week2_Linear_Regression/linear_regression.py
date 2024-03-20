import numpy as np

def MSE(x, t):
    return np.sum((x - t) ** 2, axis=0) / len(x)

class Linear_Regression:
    def __init__(self, x, weight, bias, learning_rate):
        self.weight = weight
        self.bias = bias
        self.lr = learning_rate
        self.loss = None
        self.x = x
        self.grad_weight = None
        self.grad_bias = None
        
    def get_MSE_loss(self, x, t):
        self.loss = MSE(x, t)
    
    def gradient(self, y_hat, t):
        error = y_hat - t
        self.grad_weight = np.sum(error * self.x, axis=0) / len(y_hat)
        self.grad_bias = np.sum(error, axis = 0) / len(y_hat)
    
    def update(self):
        self.weight -= self.lr * self.grad_weight
        self.bias -= self.lr * self.grad_bias