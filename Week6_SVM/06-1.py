#2018112571 김수성
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from yellowbrick.contrib.classifier import DecisionViz
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from yellowbrick.contrib.classifier import DecisionViz

# 데이터 불러오기
path = "/content/drive/MyDrive/기계학습시스템설계/twisted_data.csv"
df = pd.read_csv(path)

# y=0과 y=1로 데이터 분리
data_y0 = df[df['y'] == 0]
data_y1 = df[df['y'] == 1]

# 그래프 그리기
plt.scatter(data_y0['x1'], data_y0['x2'], c='blue', label='y=0')
plt.scatter(data_y1['x1'], data_y1['x2'], c='red', label='y=1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Scatter Plot of Data with y=0 and y=1')
plt.show()
plt.clf()

# 데이터 불러오기
path = "/content/drive/MyDrive/기계학습시스템설계/twisted_data.csv"
df = pd.read_csv(path)

X = df[["x1", "x2"]].to_numpy()
y = df["y"]

svm_poly = Pipeline([
    ("std", StandardScaler()),
    ("poly_inputs", PolynomialFeatures(degree=5)),
    ("lsmv", LinearSVC(C=1, loss="hinge"))
    ])

svm_poly.fit(X, y)
viz = DecisionViz(svm_poly)
viz.fit(X, y)
viz.draw(X, y)
