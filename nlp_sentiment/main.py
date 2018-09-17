# -*- coding=utf-8 -*-
import time
import pickle
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from numpy import *

a = time.time()
slf = open('seq_list.pickle', 'rb')
wdf = open('word_dict.pickle', 'rb')
lbf = open('labels.pickle', 'rb')
trf = open('train.pickle', 'rb')
tef = open('test.pickle', 'rb')
tenf = open('testno.pickle', 'rb')

word_dict = pickle.load(wdf)
start = time.time()
train_data = pickle.load(trf)
print('load data cost ', time.time()-start)

label_data = pickle.load(lbf)
test_data = pickle.load(tef)
test_no = pickle.load(tenf)

input_len = len(word_dict)
seq_len = len(label_data)

labels = np_utils.to_categorical(array(label_data), 5)

model = Sequential()
model.add(Dense(units=int(input_len/2), input_shape=(1, input_len)))
model.add(Activation('relu'))
model.add(Dense(units=int(input_len/4)))
model.add(Activation('relu'))
model.add(Dense(units=5))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
tr = train_data.reshape(len(train_data), 1, len(train_data[0]))
la = labels.reshape(len(labels), 1, len(labels[0]))
te = test_data.reshape(len(test_data), 1, len(test_data[0]))

model.fit(tr, la, batch_size=5120, epochs=10, verbose=1)
pred = model.predict_classes(te)
print(pred)
'''
pred = model.predict(te, batch_size=1024)

f_out = open('pred.csv', 'w')
header = "PhraseId,Sentiment\n"
f_out.write(header)
for i in range(len(pred)):
  result = argmax(pred[i])
  out_string = str(test_no[i]) + ',' + str(result) + '\n'
  f_out.write(out_string)

f_out.close()
'''
