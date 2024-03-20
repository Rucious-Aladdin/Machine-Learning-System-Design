#2018112571 김수성

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 05 - 1 - (1)

# 경로 변경 부탁드림다~
path = "/content/drive/MyDrive/기계학습시스템설계/nonlinear.csv"
data = pd.read_csv(path)

X = data["x"].to_numpy().reshape(-1, 1)
y = data["y"].to_numpy()

poly_3 = PolynomialFeatures(degree=3)
X_3 = poly_3.fit_transform(X)

poly_30 = PolynomialFeatures(degree=30)
X_30 = poly_30.fit_transform(X)

poly_300 = PolynomialFeatures(degree=300)
X_300 = poly_300.fit_transform(X)

poly_3000 = PolynomialFeatures(degree=3000)
X_3000 = poly_3000.fit_transform(X)

# Linear Regression 모델 학습
lin_model_3 = LinearRegression()
lin_model_3.fit(X_3, y)
y_3 = lin_model_3.predict(X_3)

lin_model_30 = LinearRegression()
lin_model_30.fit(X_30, y)
y_30 = lin_model_30.predict(X_30)

lin_model_300 = LinearRegression()
lin_model_300.fit(X_300, y)
y_300 = lin_model_300.predict(X_300)

lin_model_3000 = LinearRegression()
lin_model_3000.fit(X_3000, y)
y_3000 = lin_model_3000.predict(X_3000)

# 서브플롯 그리기
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

x_range = np.linspace(0, 1, 100).reshape(-1, 1)
x_range_poly_3 = poly_3.transform(x_range)
y_3_range = lin_model_3.predict(x_range_poly_3)

axs[0].scatter(X, y)
axs[0].plot(x_range, y_3_range, color='red')
axs[0].set_title('Degree 3')

x_range_poly_30 = poly_30.transform(x_range)
y_30_range = lin_model_30.predict(x_range_poly_30)

axs[1].scatter(X, y)
axs[1].plot(x_range, y_30_range, color='red')
axs[1].set_title('Degree 30')

x_range_poly_300 = poly_300.transform(x_range)
y_300_range = lin_model_300.predict(x_range_poly_300)

axs[2].scatter(X, y)
axs[2].plot(x_range, y_300_range, color='red')
axs[2].set_title('Degree 300')

x_range_poly_3000 = poly_3000.transform(x_range)
y_3000_range = lin_model_3000.predict(x_range_poly_3000)

axs[3].scatter(X, y)
axs[3].plot(x_range, y_3000_range, color='red')
axs[3].set_title('Degree 3000')

# x축 범위 설정
for ax in axs:
    ax.set_xlim([0, 1])

# 레이아웃 조정
plt.tight_layout()
plt.show()
plt.clf()

###########################################################################
#05-1 - (2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

poly_3 = PolynomialFeatures(degree=3)
X_train_3 = poly_3.fit_transform(X_train)
X_test_3 = poly_3.transform(X_test)

lin_model_3 = LinearRegression()
lin_model_3.fit(X_train_3, y_train)
y_pred_3 = lin_model_3.predict(X_test_3)

lin_model_linear = LinearRegression()
lin_model_linear.fit(X_train, y_train)
y_pred_linear = lin_model_linear.predict(X_test)

mse_3 = mean_squared_error(y_test, y_pred_3)
mse_linear = mean_squared_error(y_test, y_pred_linear)

plt.figure(figsize=(12, 6))

x_sorted = np.sort(X_test[:, 0])
y_pred_3_sorted = lin_model_3.predict(poly_3.transform(x_sorted.reshape(-1, 1)))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='red', label='Test')
plt.plot(x_sorted, y_pred_3_sorted, color='black', label='Prediction', linewidth=5.0)
plt.text(0.7, 1.5, 'MSE: {:.4f}'.format(mse_3))
plt.legend()

plt.subplot(1, 2, 2)
x_sorted = np.sort(X_test[:, 0])
y_pred_linear_sorted = lin_model_linear.predict(x_sorted.reshape(-1, 1))

plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='red', label='Test')
plt.plot(x_sorted, y_pred_linear_sorted, color='black', label='Prediction', linewidth=5.0)
plt.text(0.7, 1.5, 'MSE: {:.4f}'.format(mse_linear))
plt.legend()

plt.tight_layout()
plt.show()
plt.clf()
