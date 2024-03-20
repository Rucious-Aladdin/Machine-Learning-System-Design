import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linear_regression import MSE, Linear_Regression

## 2-1의 (1)번

# 데이터 불러오기 path를 바꿔서 넣어주세요.
path = "C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\ML_Design\\HW02\\02-1_dataset.csv"
df = pd.read_csv(path)
matrix = df.to_numpy()
column = matrix[:, 1]
x = column.T
c = np.random.randn(len(x))
y = 5 * x + 50 * np.random.randn(len(x))


#initial weight, bias
w = 0.5
b = 0.5
y_hat = x * w + b

mse = MSE(y, y_hat)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.grid(True)
plt.scatter(x, y, marker="s", color ="blue")
plt.text(10, 450, "MSE: " + str(mse))
plt.plot(x, y_hat, color="red")

#02-(2)번
lr = 1e-4
model = Linear_Regression(x, w, b, lr)

iterations = 100
for i in range(iterations):
    y_hat = x * model.weight + model.bias
    model.gradient(y_hat, y)
    model.update()
    print(model.weight, model.bias)

y_hat = model.weight * x + model.bias
mse = MSE(y, y_hat)    
plt.subplot(132)
plt.grid(True)
plt.scatter(x, y, marker="s", color ="blue")
plt.text(10, 450, "MSE: " + str(mse))
plt.plot(x, y_hat, color="red")

#02-1 (3)번

from sklearn import linear_model

reg = linear_model.LinearRegression()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
fit = reg.fit(x, y)
print(fit.coef_)
print(fit.intercept_)

y_pred = x * fit.coef_ + fit.intercept_
mse = MSE(y, y_pred)
plt.subplot(133)   
plt.scatter(x, y, marker="s", color="blue")
plt.grid(True)
plt.text(10, 450, "MSE: " + str(mse[0]))
plt.plot(x, y_pred, color="red")


plt.show()

