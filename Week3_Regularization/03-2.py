#03-2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#03-2 (1)번

#path를 바꿔주세요.
path = "/content/drive/MyDrive/기계학습시스템설계/03-2_dataset.csv"
df = pd.read_csv(path)
df.head()

y = df["X"].to_numpy()
x = range(len(y))

plt.figure(figsize = (10, 5))
plt.subplot(121)
plt.plot(x, y)

plt.subplot(122)
plt.hist(y, bins=100)

plt.tight_layout()
plt.show()
plt.clf()

#03-(2)번

max = np.max(y)
min = np.min(y)
print("최솟값: " + str(min))
print("최댓값: " + str(max))
maxmin_y = (y - min) / (max - min)

my_min = np.min(maxmin_y)
my_max = np.max(maxmin_y)
print("정규화 후 최솟값: " + str(my_min))
print("정규화 후 최댓값: " + str(my_max))

#sklearn 이용
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
y = df
scaler.fit(y)
sk_y = scaler.transform(y)

sk_y = sk_y[:, 1]
sk_min = np.min(sk_y)
sk_max = np.max(sk_y)
print("sklearn 정규화 후 최솟값: " + str(sk_min))
print("sklearn 정규화 후 최댓값: " + str(sk_max))

#03-(3)번
from sklearn.preprocessing import StandardScaler
def get_std(y): 
  mean = np.mean(y)
  std = np.sqrt(np.var(y))

  std_y = (y - mean) / std
  return std_y, mean, std


y = df["X"].to_numpy()
y, mean, std = get_std(y)

print("원본 데이터 평균: " + str(mean))
print("원본 데이터 표준편차: " + str(std))

print("표준화된 데이터 평균: " + str(float(np.mean(y))))
print("표준화된 데이터 표준편차: " + str(float(np.sqrt(np.var(y)))))

scaler = StandardScaler()
y = df

scaler.fit(y)
sk_y = scaler.transform(y)
sk_y = sk_y[:, 1]
print("sklearn 표준화된 데이터 평균: " + str(float(np.mean(sk_y))))
print("sklearn 표준화된 데이터 표준편차: " + str(float(np.sqrt(np.var(sk_y)))))

# 03-(4)번
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

randn = np.random.randn

x = randn(500)
y = -2 * x  +  1.2 * randn(500)
y = y.reshape(1, -1)
x = x.reshape(1, -1)

#reggularization
y_r = preprocessing.minmax_scale(y.T).T

x_train = x[:, :400]
t_train = y_r[:, :400]

x_test =  x[:, 400:]
t_test =  y_r[:, 400:]

x_train = x_train.reshape(-1, 1)
t_train =t_train.reshape(-1, 1)
x_test, t_test = x_test.reshape(-1, 1), t_test.reshape(-1, 1)

model = LinearRegression()

scifit = model.fit(x_train, t_train)

x1 = np.array([-4, 4])
y1 = model.coef_[0] * x1 + model.intercept_[0]

print(model.coef_[0][0], model.intercept_[0])
y_hat = model.coef_[0] * x_test + model.intercept_[0]

mse = np.sum((y_hat - t_test) ** 2, axis = 0) / len(y_hat)
print(mse)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.text(1, 0.8, "MSE: " + str(mse))
plt.title("Regularization")
plt.scatter(x_train, t_train, color="blue")
plt.scatter(x_test, t_test, color="red")
plt.plot(x1, y1, color="orange")

#Normalization
scaler = StandardScaler()
y_s = scaler.fit_transform(y.T).T

x_train = x[:, :400]
t_train = y_s[:, :400]

x_test =  x[:, 400:]
t_test =  y_s[:, 400:]

x_train = x_train.reshape(-1, 1)
t_train =t_train.reshape(-1, 1)
x_test, t_test = x_test.reshape(-1, 1), t_test.reshape(-1, 1)

model2 = LinearRegression()

scifit = model2.fit(x_train, t_train)
x1 = np.array([-4, 4])
y1 = model2.coef_[0] * x1 + model2.intercept_[0]
y_hat = model2.coef_[0] * x_test + model2.intercept_[0]
mse = np.sum((y_hat - t_test) ** 2, axis = 0) / len(y_hat)
print(mse)


plt.subplot(122)
plt.title("Normalization")
plt.text(1, 2, "MSE: " + str(mse))
plt.scatter(x_train, t_train, color="blue")
plt.scatter(x_test, t_test, color="red")
plt.plot(x1, y1, color="orange")
plt.show()
