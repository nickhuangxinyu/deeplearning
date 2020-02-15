from rl_class import *
import tensorflow

class dqn(rl):
  def __init__(self, n_actions, n_states, n_features, n_copys):
    self.n_actions = n_actions
    self.n_states = n_states
    self.n_features = n_features
    self.n_copys = n_copys
    self.train_count = 0
    self.memory_count = 0
    self.memory_size = 1024
    self.batch_size = 32
    self.memory_pool = np.zeros((self.memory_size, n_features * 2 + 2))
    self.sess = tf.Session()
    init()
    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    self.copy_from_target = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
    self.sess.run(tf.global_variables_initializer())

  def get_memory_batch(self, size):
    if self.memory_count < self.batch_size:
      print('no enough memory %d %d' % (self.memory_count, self.batch_size))
      sys.exit()
    memory_indexset = self.memory_count if self.memory_count < self.memory_size else self.memory_size
    return self.memory_pool[np.random.choice(self.memory_indexset, size=self.batch_size)]

  def init(self):
    self.s = tf.placeholder([None, self.n_features], dtype = tf.float32)
    self.s_ = tf.placeholder([None, self.n_features], dtype = tf.float32)
    with tf.variable_scope('eval_net'):  # for update: change for every time
      c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
      w1 = tf.variable([self.n_features, self.n_states], dtype = tf.float32)
      b1 = tf.variable([1, l1_hidden], dtype = tf.float32)
      l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
      w2 = tf.variable([l1_hidden, self.n_actions], dtype = tf.float32)
      b2 = tf.variable([1, self.n_actions], dtype = tf.float32)
      self.eval_net = tf.matmul(l1, w2) + b2
      
    with tf.variable_scope('target_net'):  # for label: not change frequently
      c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
      w1 = tf.variable([self.n_features, self.n_states], dtype = tf.float32)
      b1 = tf.variable([1, l1_hidden], dtype = tf.float32)
      l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
      w2 = tf.variable([l1_hidden, self.n_actions], dtype = tf.float32)
      b2 = tf.variable([1, self.n_actions], dtype = tf.float32)
      self.target_net = tf.matmul(l1, w2) + b2

  def step(self, action, current):
  def __update__(self):

  def store_transition(self, s, a, r, s_):
    # replace the old memory with new memory
    self.memory[self.memory_count % self.memory_size, :] = np.hstack((s, [a, r], s_))
    self.memory_count += 1

  def choose_action(self, input_state):
    input_state = input_state[np.newaxis, :]
    if np.random.uniform() < self.epsilon:
        # forward feed the input_state and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: input_state})
        return np.argmax(actions_value)
    else:
        return np.random.randint(0, self.n_actions)

  def learn(self):
    if self.train_count %  self.n_copys == 0:
      self.sess.run(self.copy_from_target)
      print('\ntarget_params_replaced\n')
    memory_batch = get_memory_batch(self.batch_size)
    q_next, q_eval = self.sess.run([self.target_net, self.eval_net],
    feed_dict={self.s_: batch_memory[:, -self.n_features:],  # fixed params-last_state
               self.s: batch_memory[:, :self.n_features],})  # newest params-next_state
