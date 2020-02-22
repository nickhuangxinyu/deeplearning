import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from market_snapshot import *
import tensorflow.keras.backend as K
import math
import sys
import os
import random
from tensorflow.keras.utils import to_categorical
from Dater import *

def precision_pos(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.cast(K.round(y_true * y_pred)==1, dtype='float32'))
    predicted_positives = K.sum(K.cast(K.round(y_pred)==1, dtype='float32'))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def precision_neg(y_true, y_pred):
    true_neg = K.sum(K.cast(K.round(y_true * y_pred)==4, dtype='float32'))
    predicted_neg = K.sum(K.cast(K.round(y_pred)==2, dtype='float32'))
    print(true_neg)
    print(predicted_neg)
    precision_neg = true_neg / (predicted_neg + K.epsilon())
    return precision_neg

def precision_neu(y_true, y_pred):
    true_neu = K.sum(K.cast(K.round(y_true * y_pred)==0, dtype='float32'))
    predicted_neu = K.sum(K.cast(K.round(y_pred)==0, dtype='float32'))
    print(true_neu)
    print(predicted_neu)
    precision_neu = true_neu / (predicted_neu + K.epsilon())
    return precision_neu

class LSTM:
  def __init__(self, time_step, n_input_features, n_output_features, n_lstm_units, n_lstm_next_size, batch_size):
    self.loss_history = []
    self.time_step = time_step
    self.n_input_features = n_input_features
    self.n_output_features = n_output_features
    self.n_lstm_units = n_lstm_units
    self.n_lstm_next_size =  n_lstm_next_size
    self.batch_size = batch_size
    self.build_model()

  def forward(self, x):
    return self.model.predict(x)

  def build_model(self):
    self.model = keras.Sequential([
        layers.LSTM(self.n_lstm_units, input_shape=(self.time_step, self.n_input_features)),
        layers.Dense(self.n_lstm_next_size, activation='relu'),
        layers.Dense(int(self.n_lstm_next_size/2), activation='relu'),
        layers.Dense(int(self.n_lstm_next_size/4), activation='relu'),
        layers.Dense(self.n_output_features)
        #layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        #layers.LSTM(32, return_sequences=True),
        #layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])

    self.model.compile(optimizer=keras.optimizers.Adam(0.01),
                 loss=keras.losses.CategoricalCrossentropy(), metrics=[precision_pos, precision_neg, precision_neu])

    self.model.summary()

  def train(self, x, y):
    history = self.model.fit(x, y)
    #print(y)
    #print(self.forward(x))

class DataLoader:
  def __init__(self, time_step):
    self.time_step = time_step
    self.shot = MarketSnapshot()
    self.long_neg_index = {}
    self.long_pos_index = {}
    self.long_neu_index = {}
    self.short_neg_index = {}
    self.short_pos_index = {}
    self.short_neu_index = {}
    self.datelist = []
    self.dataset = {}

  def load_one_train(self, date, ticker):
    file_path = '/home/huangxy/label/'+date+'/'+ticker+'.csv'
    if not os.path.exists(file_path):
      print(file_path + ' not existed')
      return
    df = pd.read_csv(file_path, header=None)
    self.datelist.append(date)
    ticker_columns = []
    for i in range(5):
      ticker_columns.append('bids%d' %(i))
      ticker_columns.append('asks%d' %(i))
      ticker_columns.append('bid_size%d' %(i))
      ticker_columns.append('ask_size%d' %(i))
    ticker_columns += ['last_price', 'volume', 'turnover', 'totalbuy', 'totalsell',"null"]
    df.columns = ['index', 'long', 'short'] + ticker_columns
    df=df.drop(['index', 'null'], axis=1)
    df = df.apply(lambda x:x.diff(1))
    self.dataset[date] = df.values
    df_len = len(df)
    temp = df[df['long'] ==2].index.tolist()
    self.long_neg_index[date] = list(filter(lambda x:df_len-self.time_step>x>self.time_step, temp))
    temp = df[df['long'] ==1].index.tolist()
    #print('pos label count is %d' % (len(temp)))
    self.long_pos_index[date] = list(filter(lambda x:df_len-self.time_step>x>self.time_step, temp))
    temp = df[df['long'] ==0].index.tolist()
    self.long_neu_index[date] = list(filter(lambda x:df_len-self.time_step>x>self.time_step, temp))

    temp = df[df['short'] ==2].index.tolist()
    self.short_neg_index[date] = list(filter(lambda x:df_len-self.time_step>x>self.time_step, temp))
    temp = df[df['short'] ==1].index.tolist()
    self.short_pos_index[date] = list(filter(lambda x:df_len-self.time_step>x>self.time_step, temp))
    temp = df[df['short'] ==0].index.tolist()
    self.short_neu_index[date] = list(filter(lambda x:df_len-self.time_step>x>self.time_step, temp))
    print('%s@%s loaded' % (ticker, date))

  def load_train(self, start_date, end_date, ticker):
    dl = dateRange(start_date, end_date)
    for d in dl:
      self.load_one_train(d, ticker)

  def load_one_test(self, date, ticker):
    pass

  def load_test(self, start_date, end_date, ticker):
    pass

  def get_batch(self, batch_size=128):
    date_len = min(math.ceil(math.sqrt(batch_size)), len(self.datelist))
    everyday_len = math.ceil(batch_size/date_len)
    sample_date = random.sample(self.datelist, date_len)
    samples = []
    for d in sample_date:
      index = random.sample(range(len(self.time_step, self.dataset[d])-self.time_step), everyday_len)
      for i in index:
        samples.append(self.dataset[d][i-self.time_step+1:i+1])
    samples = samples[:batch_size]
    return samples[:, :, 2:], to_categorical(samples[:, -1, 0])

  def get_sample_batch(self, batch_size):
    date_len = min(math.ceil(math.sqrt(batch_size)), len(self.datelist))
    everyday_len = math.ceil(batch_size/date_len)
    sample_date = random.sample(self.datelist, date_len)
    neu_sample=[]
    pos_sample=[]
    neg_sample=[]
    res_sample=[]
    for d in sample_date:
      num_neu = int(everyday_len*0.1)
      num_pos = int(everyday_len*0.8)
      num_neg = int(everyday_len*0.1)
      #print(len(self.long_neg_index[d]))
      #print(num_neg)
      l_neg_index = random.sample(self.long_neg_index[d], min(num_neg, len(self.long_neg_index[d])))
      for i in l_neg_index:
        neg_sample.append(self.dataset[d][i-self.time_step+1:i+1])
      #print(len(self.long_neu_index[d]))
      #print(num_neu)
      l_neu_index = random.sample(self.long_neu_index[d], min(num_neu, len(self.long_neu_index[d])))
      for i in l_neu_index:
        neu_sample.append(self.dataset[d][i-self.time_step+1:i+1])
      #print(len(self.long_pos_index[d]))
      #print(num_pos)
      l_pos_index = random.sample(self.long_pos_index[d], min(num_pos, len(self.long_pos_index[d])))
      for i in l_pos_index:
        pos_sample.append(self.dataset[d][i-self.time_step+1:i+1])
      num_res = everyday_len - len(l_pos_index) - len(l_neg_index) - len(l_neu_index)
      #print(sample_date)
      #print(num_res)
      #print(len(self.long_neu_index[sample_date[-1]]))
      if len(self.long_neu_index[sample_date[-1]]) > num_res:
        res_index = random.sample(self.long_neu_index[sample_date[-1]], num_res)
      elif len(self.long_pos_index[sample_date[-1]]) > num_res:
        res_index = random.sample(self.long_pos_index[sample_date[-1]], num_res)
      elif len(self.long_neg_index[sample_date[-1]]) > num_res:
        res_index = random.sample(self.long_neg_index[sample_date[-1]], num_res)
      for i in res_index:
        neu_sample.append(self.dataset[sample_date[-1]][i-self.time_step+1:i+1])
    #print('shape is %s %s %s %s' % (np.shape(neu_sample), np.shape(pos_sample), np.shape(neg_sample), np.shape(res_sample)))
    sample = neu_sample + pos_sample + neg_sample + res_sample
    random.shuffle(sample)
    sample = np.array(random.sample(sample, batch_size))
    #print(sample[:, :, 2:])
    #print(sample[:, -1, 0])
    return sample[:, :, 2:], to_categorical(sample[:, -1, 0])

def Evaluate(pred_y, y):
  #recall =  / (pred_y == 1).sum()
  pass

time_step=100
n_input_features=25
n_output_features=3
n_lstm_units=100
n_lstm_next_size=50
batch_size = 512
epochs = 1000

if __name__=='__main__':
  #lstm = LSTM(time_step=2, n_input_features=3, n_output_features=2, n_lstm_units=4, n_lstm_next_size=5)
  lstm = LSTM(time_step, n_input_features, n_output_features, n_lstm_units, n_lstm_next_size, batch_size)
  dl = DataLoader(time_step)
  #dl.load_one_train('20190114', '603583.SH')
  #dl.load_train('20190101', '20190115', '603583.SH')
  dl.load_train('20190101', '20190531', '002142.SZ')
  
  #x, y = dl.get_batch(32)
  #print(np.shape(x))
  #print(np.shape(y))
  for _ in range(epochs):
    x, y = dl.get_batch(batch_size)
    #print(np.shape(x))
    #print(np.shape(y))
    #sys.exit(1)
    #x = np.reshape([i for i in range(batch_size*time_step*n_input_features)], (batch_size, time_step, n_input_features))
    #y = np.reshape([i for i in range(batch_size*n_output_features)], (batch_size, n_output_features))
    lstm.train(x, y)

  # predict
  #test_x, test_y = dl.load_test()
  #pred_y = lstm.forward(test_x)
  #Evaluate(pred_y, test_y)
