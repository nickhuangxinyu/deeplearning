from sklearn import preprocessing 
import numpy as np

# 将资料分割成train与test的模块
from sklearn.model_selection import train_test_split

# 生成适合做classification资料的模块
from sklearn.datasets.samples_generator import make_classification 

# Support Vector Machine中的Support Vector Classifier
from sklearn.svm import SVC 

# 可视化数据的模块
import matplotlib.pyplot as plt 

X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2, 
    random_state=22, n_clusters_per_class=1, 
    scale=100)
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
sv = clf.support_vectors_

#可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(sv[:, 0], sv[:, 1], marker='x')
plt.show()
