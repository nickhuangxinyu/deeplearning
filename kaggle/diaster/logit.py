from handle_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print('acc is ', (lr.predict(X_test)==y_test).mean())

output['Survived'] = lr.predict(test_data)
output.to_csv('out.csv', index=False)
