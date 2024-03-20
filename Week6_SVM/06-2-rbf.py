#2018112571 김수성
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from yellowbrick.contrib.classifier import DecisionViz
import matplotlib.pyplot as plt

# 데이터 불러오기
path = "/content/drive/MyDrive/기계학습시스템설계/twisted_data.csv"
df = pd.read_csv(path)

X = df[["x1", "x2"]].to_numpy()
y = df["y"]

kernels = ['rbf']
for kernel in kernels:
    if kernel == 'poly':
        svm_model = SVC(kernel='poly', C=10, degree=3)
    elif kernel == 'rbf':
        svm_model = SVC(kernel='rbf', C=10)
    elif kernel == 'sigmoid':
        svm_model = SVC(kernel='sigmoid', C=10)

    # 데이터 표준화
    X_std = StandardScaler().fit_transform(X)

    # SVM 모델 학습
    svm_model.fit(X_std, y)

    # 시각화
    viz = DecisionViz(svm_model)
    viz.fit(X_std, y)
    viz.draw(X_std, y)