import numpy as np
import tensorflow as tf
import os
import random
from collections import defaultdict
from math import *
from keras.layers import dot
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Lambda
from keras.optimizers import SGD, Adam
from keras.layers import Embedding
from keras.layers import Multiply, Input, Flatten, multiply, subtract
from keras import regularizers
from keras.models import Model
from keras import backend as K

def load_data(data_path):
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    print ("max_u_id:", max_u_id)
    print ("max_i_id:", max_i_id)
    return max_u_id, max_i_id, user_ratings
    

data_path = os.path.join('ml-100k', 'u.data')
user_count, item_count, user_ratings = load_data(data_path)
def generate_test(user_ratings):
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test

user_ratings_test = generate_test(user_ratings)

def generate_train_batch(user_ratings=user_ratings, user_ratings_test=user_ratings_test, item_count=item_count, batch_size=512):
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]
        
        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])
    return np.asarray(t)

def generate_test_batch(user_ratings=user_ratings, user_ratings_test=user_ratings_test, item_count=item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count+1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield np.asarray(t)

def gen_test():
  t = []
  u_array = []
  view_array = []
  unview_array = []
  for u in user_ratings.keys():
    i = user_ratings_test[u]
    for j in range(1, item_count+1):
      if not (j in user_ratings[u]):
        u_array.append(u)
        view_array.append(i)
        unview_array.append(j)
  return [np.array(u_array), np.array(view_array), np.array(unview_array)]
