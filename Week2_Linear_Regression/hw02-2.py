from sklearn import linear_model
import numpy as np

reg1 = linear_model.LinearRegression()
x = np.array([130, 250, 190, 300, 210, 220, 170])
x = x.reshape(-1, 1)
y = np.array([16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2])
y = y.reshape(-1, 1)

fit = reg1.fit(x, y)

#02-2 (1), (2)번
print("계수: " + str(fit.coef_))
print("절편: " + str(fit.intercept_))
print('결정계수: ', fit.score(x.reshape(-1,1), y.reshape(-1,1)))
print("270마력의 예상연비: " + str(float(fit.predict([[270]]))) + " km/l\n")

#02-2 (3)번

x1 = np.array([130, 250, 190, 300, 210, 220, 170])
x1 = x1.reshape(-1, 1)
x2 = np.array([1900, 2600, 2200, 2900, 2400, 2300, 2100])
x2 = x2.reshape(-1, 1)
y = np.array([16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2])
y = y.reshape(-1, 1)

reg2 = linear_model.LinearRegression()
fit2 = reg2.fit(np.c_[x1, x2], y)
print("계수: " + str(fit2.coef_))
print("절편: " + str(fit2.intercept_))
print('결정계수: ', fit2.score(np.c_[x1, x2], y.reshape(-1,1)))
print("270마력의 예상연비: " + str(float(fit2.predict([[270, 2500]]))) + " km/l")