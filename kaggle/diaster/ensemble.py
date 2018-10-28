from handle_data import *

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
import warnings
from sklearn.svm import SVC
warnings.filterwarnings('ignore')

#X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=0)
svc = SVC()
svc_param = {'kernel':['poly'],# 'poly'],
             'C':[1, 3, 10]
             }

gridsvc = GridSearchCV(svc, param_grid = svc_param, scoring="accuracy", verbose = 1)

gridsvc.fit(train_data, labels)
print(gridsvc.best_estimator_)

model = XGBClassifier(
    nthread=24)

model_param = {'n_estimators':[3,5,10, 30,50],
               'max_depth':[3,5,9],
               'min_child_weight':[2,4,6]} #,
#'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
# 'subsample':[0.2, 0.5, 0.7, 1.0],
#              'lambda':[0.1,0.2,0.3, 1.0]}

gridmodel = GridSearchCV(model, param_grid = model_param, scoring="accuracy", verbose = 1)

gridmodel.fit(train_data, labels)

print(gridmodel.best_estimator_)


votingC = VotingClassifier(estimators=[('svc', gridsvc.best_estimator_), ('xgb', gridmodel.best_estimator_)])

votingC.fit(train_data, labels)

for clf, label in zip([gridsvc.best_estimator_, gridmodel.best_estimator_, votingC], ['svc', 'xgb', 'Ensemble']):
        scores = cross_val_score(clf, train_data, labels, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

'''
output['Survived'] = model.predict(test_data)
output.to_csv('out.csv', index=False)
scores = cross_val_score(model, train_data, labels, cv=10, scoring='accuracy')
print(scores.mean())
'''
