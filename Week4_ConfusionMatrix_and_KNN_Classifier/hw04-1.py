#2018112571 김수성
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


#과제 4-1

dacs = [[75, 24], [77, 29], [83, 19], [81, 32], [73, 21], [99, 22], [72, 19], [83, 34]]
samo = [[76, 55], [78, 58], [82, 53], [88, 54], [76, 61], [83, 52], [81, 57], [89, 64]]
malt = [[35, 23], [39, 26], [38, 19], [41, 30], [30, 21], [57, 24], [41, 28], [35, 20]]

d_data = np.array(dacs)
d_label = np.zeros(len(d_data))
s_data = np.array(samo)
s_label = np.ones(len(s_data))
m_data = np.array(malt)
m_label = 2 * np.ones(len(m_data))

# 4-1 (1번)
print("닥스훈트(0): " + str(dacs))
print("사모예드(1): " + str(samo))
print("말티즈(2): " + str(malt))

# 4-1 (2번)

dogs = np.concatenate((d_data, s_data))
dogs = np.concatenate((dogs, m_data))
labels = np.concatenate((d_label, s_label))
labels = np.concatenate((labels, m_label))

print(dogs.shape)
print(labels.shape)

dogs_dict = {0:"닥스훈트", 1:"사모예드", 2:"말티즈"}

k = 3
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(dogs, labels)

y_pred_all = knn.predict(dogs)
conf_mat = confusion_matrix(labels, y_pred_all)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(conf_mat, cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(122)
plt.imshow(conf_mat, cmap=plt.cm.gray)
plt.colorbar()
ConfusionMatrixDisplay(confusion_matrix=conf_mat).plot()

# 4-1 (3번)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def custom_metrics(labels, predictions):
    tp = sum((labels == 1) & (predictions == 1))
    fp = sum((labels == 0) & (predictions == 1))
    fn = sum((labels == 1) & (predictions == 0))
    tn = sum((labels == 0) & (predictions == 0))
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, accuracy, f1_score

dm_label = 0.5 * np.concatenate((labels[:8], labels[16:24])).astype("int32")
dm_pred = 0.5 * np.concatenate((y_pred_all[:8], y_pred_all[16:24])).astype("int32")

print(dm_label)
print(dm_pred)

precision, recall, accuracy, f1 = custom_metrics(dm_label, dm_pred)
print("내가만든 Precision: ", precision)
print("내가만든 Recall: ", recall)
print("내가만든 Accuracy: ", accuracy)
print("내가만든 F1-score: ", f1)

print("scikit-learn의 Precision: " , precision_score(dm_label, dm_pred))
print("scikit-learn의 Recall: ", recall_score(dm_label, dm_pred))
print("scikit-learn의 accuracy: ", accuracy_score(dm_label, dm_pred))
print("scikit-learn의 F1-score: ", f1_score(dm_label, dm_pred))

# 4-1 (4번)

k = 3
knn3 = KNeighborsClassifier(n_neighbors = k)
knn3.fit(dogs, labels)

k = 5
knn5 = KNeighborsClassifier(n_neighbors = k)
knn5.fit(dogs, labels)

k = 7
knn7 = KNeighborsClassifier(n_neighbors = k)
knn7.fit(dogs, labels)

data = np.array([[[58, 30]], [[80, 26]], [[80, 41]], [[75, 55]]])

A_3 = knn3.predict(data[0])
A_5 = knn5.predict(data[0])
A_7 = knn7.predict(data[0])

B_3 = knn3.predict(data[1])
B_5 = knn5.predict(data[1])
B_7 = knn7.predict(data[1])

C_3 = knn3.predict(data[2])
C_5 = knn5.predict(data[2])
C_7 = knn7.predict(data[2])

D_3 = knn3.predict(data[3])
D_5 = knn5.predict(data[3])
D_7 = knn7.predict(data[3])

print("A: [[58, 30]]")
print("k=3:", dogs_dict[int(A_3.tolist()[0])])
print("k=5:", dogs_dict[int(A_5.tolist()[0])])
print("k=7:", dogs_dict[int(A_7.tolist()[0])])
print()

print("B: [[80, 26]]")
print("k=3:", dogs_dict[int(B_3.tolist()[0])])
print("k=5:", dogs_dict[int(B_5.tolist()[0])])
print("k=7:", dogs_dict[int(B_7.tolist()[0])])
print()

print("C: [[80, 41]]")
print("k=3:", dogs_dict[int(C_3.tolist()[0])])
print("k=5:", dogs_dict[int(C_5.tolist()[0])])
print("k=7:", dogs_dict[int(C_7.tolist()[0])])
print()

print("D: [[75, 55]]")
print("k=3:", dogs_dict[int(D_3.tolist()[0])])
print("k=5:", dogs_dict[int(D_5.tolist()[0])])
print("k=7:", dogs_dict[int(D_7.tolist()[0])])

# 4-1 (5번)

import matplotlib.pyplot as plt

dacs = [[75, 24], [77, 29], [83, 19], [81, 32], [73, 21], [99, 22], [72, 19], [83, 34]]
samo = [[76, 55], [78, 58], [82, 53], [88, 54], [76, 61], [83, 52], [81, 57], [89, 64]]
malt = [[35, 23], [39, 26], [38, 19], [41, 30], [30, 21], [57, 24], [41, 28], [35, 20]]

data_plot = {
    'A': [[58, 30], 'pink'],
    'B': [[80, 26], 'gray'],
    'C': [[80, 41], 'skyblue'],
    'D': [[75, 55], 'forestgreen']
}

dacs = list(zip(*dacs))  # Transpose for plotting
samo = list(zip(*samo))
malt = list(zip(*malt))

plt.scatter(dacs[0], dacs[1], c='red', label='Dachshund')
plt.scatter(samo[0], samo[1], c='blue', label='Samoyed')
plt.scatter(malt[0], malt[1], c='green', label='Maltese')

for label, (point, color) in data_plot.items():
    plt.scatter(point[0], point[1], c=color, label=f'Point {label}', s=300, marker='o')


plt.title("Dog Size")
plt.xlabel('Length')
plt.ylabel('Height')

plt.legend()
plt.show()

# 4-1 (6번)
from sklearn import cluster

def kmeans_predict_plot(X, k):
  model = cluster.KMeans(n_clusters=k)
  model.fit(X)
  labels = model.predict(X)
  colors = np.array(["red", "green", "blue", "magenta"])
  plt.suptitle("k-means Clustering, k={}".format(k))
  plt.scatter(X[:, 0], X[:, 1], color=colors[labels])
  plt.show()
  return labels

data = data.reshape(4, -1)
dogs = np.concatenate((dogs, data))
print(dogs.shape)

kmeans_predict_plot(dogs, 2)
kmeans_predict_plot(dogs, 3)
kmeans_predict_plot(dogs, 4)
