from agent import *
import pandas as pd
import numpy as np
import sys
sys.path.append('/root/quant/tools/common')
from market_snapshot import *

epochs = 100
target_profit = 2
stoploss = 2
fee = 0.5

hist_ws = 10
forward_ws = 10
wallet = 0
learn_step = 200

shot = MarketSnapshot()
columns = shot.get_columns()
columns = columns[1:]

col_dict = {c:i for i, c in enumerate(columns)}

def gen_sample(file_name, hist_ws=hist_ws, forward_ws=forward_ws):
  print('running read_csv')
  df = pd.read_csv(file_name)
  df.columns = shot.get_columns()
  del df['ticker']
  print('read over')
  hist_sample, forward_sample = [], []
  for i in range(hist_ws, len(df)-forward_ws):
    hist_sample.append(df.iloc[i-hist_ws:i].values)
    forward_sample.append(df.iloc[i:i+forward_ws].values)
  print('gen_sample over')
  return hist_sample, forward_sample

# state will be last windowsize snapshot mix with current pos

def gen_reward(hist, forward, action, wallet):
  print(hist)
  reward = 0.0
  if action == 1:  # buy action
    buy_price = hist[-1][col_dict['asks[0]']]  # ask price
    if wallet == 0:  # best sitution's pnl as reward
      close_price = forward[:, col_dict['bids[0]']].max()
      wallet -= buy_price
      return close_price-buy_price-fee, wallet
    else:
      temp = wallet
      wallet = 0
      return temp-buy_price-fee, wallet
  elif action == 2: # sell action
    sell_price = hist[-1][col_dict['bids[0]']]  # bid price
    if wallet == 0:  # best sitution's pnl as reward
      close_price = forward[:, col_dict['asks[0]']].min()
      wallet += sell_price
      return sell_price-close_price-fee, wallet
    else:
      temp = wallet
      wallet = -1
      return sell_price - wallet - fee, wallet
  else:  # sit
    if wallet == 0:
      return  0.0, wallet
    else:
      return fee/forward_ws, wallet

train_files=['/running/2019-05-22/ni8888.csv']
agent = Agent()
for i in range(epochs):
  for f in train_files:
    wallet = 0
    count = 1
    hist, forward = gen_sample(f)
    state = np.hstack((hist[0],[[wallet]]*len(hist[0]))) # state defination
    for i, hs in enumerate(hist):
      h, f = hist[i], forward[i]
      action = int(agent.choose_action(np.expand_dims(state,0)))
      reward, wallet = gen_reward(h, f, action, wallet)
      s_ = np.hstack((h,[[wallet]]*len(h))) # state defination
      agent.store_memory(state, action, reward, s_)
      if count % learn_step == 0:
        agent.learn()
      state = s_
      count += 1
