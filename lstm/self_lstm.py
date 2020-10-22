import numpy as np
import time

start=time.time()

seq = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

num_sample = 100000
num_valid = 10000
sample_data_list = []
sample_y = []
for i in range(num_sample):
  r = np.random.randint(1,len(seq))
  length = np.random.randint(1, len(seq))
  temp = seq[r:r+length]
  if r+length > len(seq):
    temp += seq[0:r+length-len(seq)]
  sample_data_list.append(temp)
  sample_y.append(seq[(r+length)%len(seq)])

valid_data_list = []
valid_y = []
for i in range(num_valid):
  r = np.random.randint(1,len(seq))
  length = np.random.randint(1, len(seq))
  temp = seq[r:r+length]
  if r+length > len(seq):
    temp += seq[0:r+length-len(seq)]
  valid_data_list.append(temp)
  valid_y.append(seq[(r+length)%len(seq)])

#for i in range(num_sample):
  #print('%s->%s' %(sample_data_list[i], sample_y[i]))

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.preprocessing import sequence,text
from keras.optimizers import Adam

tokenizer = Tokenizer(num_words=len(seq)+1)
tokenizer.fit_on_texts(list(seq))

sample_data_list =[list(i) for i in sample_data_list]
X_train = tokenizer.texts_to_sequences(sample_data_list)
label_x = tokenizer.texts_to_sequences(sample_y)

valid_data_list =[list(i) for i in valid_data_list]
X_valid = tokenizer.texts_to_sequences(valid_data_list)
label_valid = tokenizer.texts_to_sequences(valid_y)

X_train = sequence.pad_sequences(X_train, maxlen=len(seq), padding='post')
X_train = X_train.reshape(len(X_train), 1, len(X_train[0]))

X_valid = sequence.pad_sequences(X_valid, maxlen=len(seq), padding='post')
X_valid = X_valid.reshape(len(X_valid), 1, len(X_valid[0]))

label_x = to_categorical([i[0] for i in label_x])
label_valid = to_categorical([i[0] for i in label_valid])

#label_x = label_x.reshape(label_x.shape[0], 1, label_x.shape[1])
#label_valid = label_valid.reshape(label_valid.shape[0], 1, label_valid.shape[1])

model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(LSTM(80))
model.add(Dense(100, activation='relu'))
#model.add(LSTM(80))
#model.add(Dense(50, activation='relu'))
model.add(Dense(len(seq)+1, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

history=model.fit(X_train, label_x, validation_data=(X_valid, label_valid), 
                  epochs=50, batch_size=1024, verbose=1)

pred = model.predict(X_valid)
a = [np.argmax(i) for i in pred]
b = [np.argmax(i) for i in label_valid]
c = [a[i] - b[i] for i in range(len(a))]
d = [c.index(i) for i in c if i != 0]
for i in d:
  print(X_valid[i])
  print(np.argmax(label_valid[i]))
  print(a[i])

print('cost', time.time()-start)
