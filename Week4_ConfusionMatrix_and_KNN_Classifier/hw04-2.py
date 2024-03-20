#2018112571 김수성

# 4-2 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def kmeans_predict_plot(X, k):
  model = cluster.KMeans(n_clusters=k)
  model.fit(X)
  labels = model.predict(X)
  colors = np.array(["red", "green", "blue", "magenta"])
  plt.suptitle("k-means Clustering, k={}".format(k))
  plt.scatter(X[:, 0], X[:, 1], color=colors[labels])
  plt.show()
  return labels

iris = load_iris()
X = iris.data

pred = kmeans_predict_plot(X, k=3)
print('iris 데이터의 군집화 정확도: %.3f%%'%(np.sum(pred == iris.target) / len(iris.target) * 100))