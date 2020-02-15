import random
import tensorflow as tf
from keras.models import Sequential
import numpy as np
import pandas as pd
import sys

class Agent:
  def __init__(self, hist_ws=100, forward_ws = 100, n_hidden = 5, n_action = 3, memory_size = 5000, batch_size = 128, epsilon = 0.9, replace_count = 100):
    self.learn_step_counter = 0
    self.shot_length = 28
    self.lr = 0.01
    self.hist_ws = hist_ws
    self.forward_ws = forward_ws
    self.n_hidden = n_hidden
    self.n_action = n_action
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.state_dim = self.hist_ws*(self.shot_length+1)
    self.pos_memory = np.zeros([self.memory_size, 2*self.state_dim + 2])
    self.neg_memory = np.zeros([self.memory_size, 2*self.state_dim + 2])
    self.epsilon = epsilon
    self.replace_count = replace_count
    self.create_model()
    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.cost_his = []

  def create_model(self):
    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
    self.s = tf.placeholder(tf.float32, [None, self.hist_ws, int(self.state_dim/self.hist_ws)])
    self.q_target = tf.placeholder(tf.float32, [None, self.n_action])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_hidden, forget_bias=1.0)
    #init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
    with tf.variable_scope('eval_net'):
      #w_eval = tf.get_variable('w_eval', [self.n_hidden*self.hist_ws, self.n_action], initializer=w_initializer)
      #self.q_eval = tf.matmul(tf.reshape(hiddens, (-1,hiddens.shape[1]*hiddens.shape[2])), w_eval) + b_eval
      #self.q_eval = tf.matmul(tf.reshape(self.s, (-1, self.s.shape[1]*self.s.shape[2])), w_eval) + b_eval
      #hiddens, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.s, initial_state=init_state, dtype=tf.float32)
      hiddens, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.s, dtype=tf.float32)
      w_eval = tf.get_variable('w_eval', [self.n_hidden, self.n_action], initializer=w_initializer)
      b_eval = tf.get_variable('b_eval', [1, self.n_action], initializer=b_initializer)
      self.q_eval = tf.matmul(hiddens[:,-1,:], w_eval)# + b_eval
    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
    with tf.variable_scope('train'):
      self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    self.s_ = tf.placeholder(tf.float32, [None, self.hist_ws, int(self.state_dim/self.hist_ws)], name='s_')
    with tf.variable_scope('target_net'):
      #hiddens, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.s_, initial_state=init_state, dtype=tf.float32)
      hiddens, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.s_, dtype=tf.float32)
      #w_target = tf.get_variable('w_eval', [self.n_hidden*self.hist_ws, self.n_action], initializer=w_initializer)
      w_target = tf.get_variable('w_eval', [self.n_hidden, self.n_action], initializer=w_initializer)
      b_target = tf.get_variable('b_eval', [1, self.n_action], initializer=b_initializer)
      self.q_target_cal = tf.matmul(hiddens[:,-1,:], w_target) + b_target
      #self.q_target_cal = tf.matmul(self.s_, w_eval) + b_eval

  def choose_action(self, observation):
    if np.random.uniform() < self.epsilon:
      # forward feed the observation and get q value for every actions
      actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
      action = np.argmax(actions_value)
    else:
      action = np.random.randint(0, self.n_action)
    return int(action)

  def learn(self):
    if self.learn_step_counter % self.replace_count == 0:
      self.sess.run(self.replace_target_op)
      print('\ntarget_params_replaced\n')
    if self.pos_memory_counter > self.memory_size:
      pos_sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
      pos_sample_index = np.random.choice(self.pos_memory_counter, size=self.batch_size)
    if self.neg_memory_counter > self.memory_size:
      neg_sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
      neg_sample_index = np.random.choice(self.neg_memory_counter, size=self.batch_size)
    neg_batch_memory = self.neg_memory[neg_sample_index, :]
    pos_batch_memory = self.pos_memory[pos_sample_index, :]
    sample_index = np.random.choice(self.batch_size*2, self.batch_size)
    all_batch_memory = np.vstack((neg_batch_memory, pos_batch_memory))
    batch_memory = all_batch_memory[sample_index, :]
    q_eval, q_target = self.sess.run([self.q_eval, self.q_target_cal], feed_dict={self.s:np.reshape(batch_memory[:,:self.state_dim], (-1, self.hist_ws, int(self.state_dim/self.hist_ws))), self.s_:np.reshape(batch_memory[:,-self.state_dim:], (-1, self.hist_ws, int(self.state_dim/self.hist_ws)))})
    action = list(map(int, batch_memory[:, self.state_dim]))
    reward = batch_memory[:, self.state_dim+1]
    batch_index = np.arange(self.batch_size, dtype=np.int32)
    q_target[batch_index, action] = reward + np.max(q_target, axis=1)
    _, self.cost = self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.s: np.reshape(batch_memory[:, :self.state_dim], (-1, self.hist_ws, self.shot_length+1)),
                                            self.q_target: q_target})
    
  def print_qeval(self, state):
    return self.sess.run(self.q_eval, feed_dict={self.s: state})

  def store_memory(self, s, a, r, s_):
    print('storing %f %d' %(r, a))
    if not hasattr(self, 'pos_memory_counter'):
      self.pos_memory_counter = 0
    if not hasattr(self, 'neg_memory_counter'):
      self.neg_memory_counter = 0
    transition = np.hstack((s.flatten(), [int(a), r], s_.flatten()))
    if r > 0:
      self.pos_memory[self.pos_memory_counter%self.memory_size] = transition
      self.pos_memory_counter += 1
    else:
      self.neg_memory[self.neg_memory_counter%self.memory_size] = transition
      self.neg_memory_counter += 1

  def real_act(self, s):
    actions_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
    return s[:, -1,:], np.argmax(actions_value)
