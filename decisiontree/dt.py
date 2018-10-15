from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('iris')
dot_data = tree.export_graphviz(clf, out_file=None, # doctest: +SKIP
                            feature_names=iris.feature_names,  # doctest: +SKIP
                            class_names=iris.target_names,  # doctest: +SKIP
                            filled=True, rounded=True,  # doctest: +SKIP
                            special_characters=True)  # doctest: +SKIP

graph = graphviz.Source(dot_data)  # doctest: +SKIP
graph
