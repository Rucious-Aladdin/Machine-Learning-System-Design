#20181126\571 김수성
import numpy as np
import matplotlib.pyplot as plt

class GD:
    def __init__(self, fx, dx, lr, x0):
        self.fx = fx
        #해석적 방식으로 미분을 구함.
        self.dx = dx
        self.grad = 0.0
        self.lr = lr
        self.x = x0
    def gradient(self):
        self.grad = self.dx(self.x)
    def update(self):
        self.x -= self.lr * self.grad
    def predict(self):
        return self.fx(self.x)
    
def fx1(x):
    return x * x
def dx1(x):
    return 2 * x
def fx2(x):
    return x * np.sin(x * x)
def dx2(x):
    return np.sin(x * x) + 2 * x * x * np.cos(x * x)

iteration = 10
x0 = 10
lr = 0.1

model = GD(fx1, dx1, lr, x0) #fx1, dx1, lr, x0

x_list = []
for i in range(iteration):
    model.gradient()
    model.update()
    print(model.x, model.predict())
    x_list.append(model.x)

x_list = np.array(x_list)
y_list = fx1(x_list)

print("")
plt.subplot(121)
x=np.arange(-10, 11, step = 1)
plt.plot(x, fx1(x), color="blue")
plt.plot(x_list, y_list, marker="s", color = "blue")
plt.xlim(-10, 10)

## model2
iteration = 10
x0 = 1.6
lr = 0.01
model = GD(fx2, dx2, lr, x0) #fx1, dx1, lr, x0


x_list = []
for i in range(iteration):
    model.gradient()
    model.update()
    print(model.x, model.predict())
    x_list.append(model.x)

x_list = np.array(x_list)
y_list = fx2(x_list)

print("")
plt.subplot(122)
x=np.arange(-3, 3, step = 0.01)
plt.plot(x, fx2(x), color="blue")
plt.plot(x_list, y_list, marker="s", color = "blue")
plt.xlim(-3, 3)
plt.tight_layout()
plt.savefig("C:/Users/IT대학_000/HW1/hw1-3.png")
