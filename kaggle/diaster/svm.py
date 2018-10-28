from handle_data import *

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=0)
clf = SVC(kernel='rbf')
clf.fit(train_data, labels)
scores = cross_val_score(clf, train_data, labels, cv=10, scoring='accuracy')

#print('acc is ', (clf.predict(X_test)==y_test).mean())
print(scores.mean())

output['Survived'] = clf.predict(test_data)
output.to_csv('out.csv', index=False)
