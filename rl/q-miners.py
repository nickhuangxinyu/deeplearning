import numpy as np
import pandas as pd
import copy
import os
import time
import sys

ACTIONS={'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
width=5
height=5
epochs = 1000

start = (0,0)
terminal = (4,4)

EPSILON = 0.6   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor

Hinder=[(1,3), (3,3)]

def getxy(s):
  return (int(s/width), s-int(s/width)*width)

def gets(x, y):
  return x*width+y;

def build_q_tables(num_actions=len(ACTIONS), num_states=width*height, default_value=0):
  return pd.DataFrame([[default_value]*num_actions]*num_states, columns=ACTIONS.keys())

def legal(x, y):
  if x >= width or x < 0 or y >= height or y < 0 or (x,y) in Hinder:
    return False
  return True

def next_state(s, action):
  (x,y) = getxy(s)
  new_x = x+ACTIONS[action][0]
  new_y = y+ACTIONS[action][1]
  if legal(new_x, new_y):
    x = new_x
    y = new_y
  return gets(x, y)

def choose_actions(s, q_table):
  actions_space = q_table.iloc[s]
  (x,y) = getxy(s)
  # eposilon greedy, 1-eposilon random
  if np.random.uniform() > EPSILON or (actions_space==0).all():
    action = np.random.choice(ACTIONS.keys())
    while next_state(s, action) == s: # out the wall
      action = np.random.choice(ACTIONS.keys())
    return action
  else:
    return actions_space.idxmax()

def interact_with_env(s, action):
  (x,y) = getxy(next_state(s, action))
  # update: if 
  if (x,y) == terminal:
    R = 1
  else:
    R = 0
  return (x,y), R

def show(s):
  time.sleep(0.05)
  os.system('clear')
  init = ['o'*width]*height
  row = init[s[0]]
  row = row[:s[1]] + '*' + row[s[1]+1:]
  init[s[0]] = row
  for h in Hinder:
    row = init[h[0]]
    row = row[:h[1]] + 'x' + row[h[1]+1:]
    init[h[0]] = row
  for i in init:
    print ' '.join(i)

def rl():
  q_table = build_q_tables()
  show(start)
  for i in range(epochs):
    count = 0
    current_state = copy.deepcopy(start)
    while current_state != terminal: 
      action = choose_actions(gets(current_state[0], current_state[1]), q_table)
      next_state, r = interact_with_env(gets(current_state[0], current_state[1]), action)
      q_real = q_table.loc[gets(current_state[0], current_state[1]), action]
      '''
      if r == 1:
        print 'next step is terminal'
        print current_state
        print action
        a = raw_input()
      '''
      if next_state != terminal:
        q_pred = r+GAMMA*q_table.loc[gets(next_state[0], next_state[1])].max()
      else:
        q_pred = r #+GAMMA*q_table.loc[gets(next_state[0], next_state[1])].max() next state is terminal no qtable for it
      q_table.loc[gets(current_state[0], current_state[1]), action] += ALPHA*(q_pred-q_real)
      current_state = next_state
      show(current_state)
      count += 1
    print str(i)+'th epoch end: cost ' + str(count) +' step'
    print q_table
    #a = raw_input()
  return q_table

def main():
  qtable = rl()
  print qtable

main()
