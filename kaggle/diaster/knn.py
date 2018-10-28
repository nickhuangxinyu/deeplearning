from handle_data import *

from sklearn.cross_validation import train_test_split
from sklearn import neighbors

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=0)
clf = neighbors.KNeighborsClassifier(2)
clf.fit(X_train, y_train)

print('acc is ', (clf.predict(X_test)==y_test).mean())

output['Survived'] = clf.predict(test_data)
output.to_csv('out.csv', index=False)
