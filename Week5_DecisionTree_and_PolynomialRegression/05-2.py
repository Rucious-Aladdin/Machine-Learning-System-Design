#2018112571 김수성

# 05-2-(1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

from sklearn.tree import plot_tree
plt.figure(figsize=(10, 8))
plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
plt.clf()

y_pred = tree_model.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print("Macro F1-score:", macro_f1)

# 05-2-(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree_model_id3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree_model_id3.fit(X_train, y_train)

plt.figure(figsize=(10, 8))
plot_tree(tree_model_id3, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

y_pred_id3 = tree_model_id3.predict(X_test)
macro_f1_id3 = f1_score(y_test, y_pred_id3, average='macro')

print("Macro F1-score :", macro_f1_id3)