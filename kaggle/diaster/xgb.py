from handle_data import *

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=0)

model = XGBClassifier(
    max_depth=6,
    n_estimators=100,
    nthread=24)


model.fit(
    X_train, 
    y_train,
    verbose=True,)


print('acc is ', (model.predict(X_test)==y_test).mean())
output['Survived'] = model.predict(test_data)
output.to_csv('out.csv', index=False)
scores = cross_val_score(model, train_data, labels, cv=10, scoring='accuracy')
print(scores.mean())
