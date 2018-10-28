import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense,Dropout
from sklearn import preprocessing

def Norm(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

train = pd.read_csv('train.csv')
train['is_train'] = [True]*len(train)
test = pd.read_csv('test.csv')
test['is_train'] = [False]*len(test)
labels = train['Survived']
del train['Survived']

concat = pd.concat([train, test], ignore_index=True)
del concat ['Name']
del concat['Ticket']
del concat['PassengerId']
del concat['Cabin']

tokenizer = Tokenizer(num_words=len(concat))
tokenizer.fit_on_texts(concat['Sex'])
concat['Sex'] = np.array(tokenizer.texts_to_sequences(concat['Sex'])).flatten()

concat['Embarked'] = np.array(concat['Embarked'], dtype=str).flatten()
tokenizer = Tokenizer(num_words=len(concat))
tokenizer.fit_on_texts(concat['Embarked'])
concat['Embarked'] = np.array(tokenizer.texts_to_sequences(concat['Embarked'])).flatten()

for i in concat:
  concat[i][np.isnan(concat[i])]=concat[i].mean()
  #concat[i] = Norm(concat[i])

train_data = concat[concat['is_train']>0]
test_data = concat[concat['is_train']==0]
del train_data['is_train']
del test_data['is_train']
output = pd.read_csv('gender_submission.csv')
output['Survived'] = [-1]*len(test_data)
