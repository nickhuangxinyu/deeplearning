from sklearn import preprocessing 
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.datasets.samples_generator import make_classification 

from sklearn.svm import SVC 

import matplotlib.pyplot as plt 

X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2, 
    random_state=22, n_clusters_per_class=1, 
    scale=100)
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
sv = clf.support_vectors_

XX, YY = np.mgrid[X[:, 0].min():X[:, 0].max(), X[:, 1].min():X[:, 1].max()]
XX, YY = np.meshgrid(np.arange(X[:, 0].min(), X[:,0].max(), 0.01), np.arange(X[:, 1].min(), X[:,1].max(), 0.6))
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
Z = clf.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'])

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(sv[:, 0], sv[:, 1], marker='x')
plt.show()
