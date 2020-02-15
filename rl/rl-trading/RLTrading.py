from agent import *
import pandas as pd
import numpy as np
import sys
sys.path.append('/root/quant/tools/common')
from market_snapshot import *
from Trader import *
import matplotlib.pyplot as plt

shot = MarketSnapshot()
columns = shot.get_columns()
columns = columns[1:]
col_dict = {c:i for i, c in enumerate(columns)}

hist_ws = 100
forward_ws = 100

class RLTrading:
  def __init__(self):
    self.agent = Agent(hist_ws=hist_ws, forward_ws=forward_ws)
    self.trader = Trader(record=True)
    self.fee_rate = 0.0001

  def Train(self, epochs=100, train_files = ['/running/2019-05-22/ni8888.csv'], target_profit = 2, stoploss = 2, fee = 0.5, learn_step = 2000):
    for _ in range(epochs):
      for f in train_files:
        wallet = 0
        count = 1
        hist, forward = self._gen_sample(f)
        state = np.hstack((hist[0],[[wallet]]*len(hist[0]))) # state defination
        for i, hs in enumerate(hist):
          h, f = hist[i], forward[i]
          action = int(self.agent.choose_action(np.expand_dims(state,0)))
          print('choose action %d' %(action))
          reward, wallet = self._cal_reward(h, f, action, wallet)
          s_ = np.hstack((h,[[wallet]]*len(h))) # state defination
          self.agent.store_memory(state, action, reward, s_)
          if count % learn_step == 0:
            self.agent.learn()
          state = s_
          count += 1

  def Plot(self, f):
    df = pd.read_csv(f)
    df.columns = shot.get_columns()
    df['mid'] = (df['asks[0]'] + df['bids[0]']) / 2
    plt.title(f)
    plt.plot(df['mid'])
    plt.show()

  def Test(self, test_file = ['/running/2019-05-22/ni8888.csv',
                              '/running/2019-05-23/ni8888.csv',
                              '/running/2019-05-24/ni8888.csv',
                              '/running/2019-05-27/ni8888.csv',]):
    for f in test_file:
      #self.Plot(f)
      wallet = 0
      count = 1
      hist, forward = self._gen_sample(f)
      for i, hs in enumerate(hist):
        h, f = hist[i], forward[i]
        state = np.hstack((h,[[wallet]]*len(h))) # state defination
        shot, action = self.agent.real_act(np.expand_dims(state,0))
        shot = shot.flatten()
        reward, wallet = self._cal_reward(h, f, action, wallet)
        if reward > 0:
          print(self.agent.print_qeval(np.expand_dims(state,0)))
        if action == 1:  # buy
          self.trader.RegisterOneTrade("ticker", 1, shot[col_dict['asks[0]']])
        elif action ==2:  # sell
          self.trader.RegisterOneTrade("ticker", 1, shot[col_dict['bids[0]']])
    print('in %d actionspace'%(len(hist)))
    self.trader.Summary()

  # gen_sample: generate sample, one by one, hist_sample->[i, i+hist_ws], forward_sample->[i, i+forward_ws]
  def _gen_sample(self, file_name, hist_ws=hist_ws, forward_ws=forward_ws):
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

  # cal_reward: reward based on action,forward and wallet, consider the best situation
  def _cal_reward(self, hist, forward, action, wallet):
    reward = 0.0
    fee = hist[-1][col_dict['asks[0]']]*self.fee_rate
    if action == 1:  # buy action
      buy_price = hist[-1][col_dict['asks[0]']]  # ask price
      if wallet == 0:  # best sitution's pnl as reward
        close_price = forward[:, col_dict['bids[0]']].max()
        wallet = 1
        return close_price-buy_price-fee, wallet
      elif wallet == -1:  # close sell, using buy
        wallet = 0
        avg_close_price_buy = forward[:, col_dict['asks[0]']].mean()
        return avg_close_price_buy-buy_price, wallet
      else:
        return -fee/forward_ws, wallet
    elif action == 2: # sell action
      sell_price = hist[-1][col_dict['bids[0]']]  # bid price
      if wallet == 0:  # best sitution's pnl as reward
        close_price = forward[:, col_dict['asks[0]']].min()
        wallet -= 1
        return sell_price-close_price-fee, wallet
      elif wallet == 1:  # close buy, using sell, reward are price-avgprice
        wallet = 0
        avg_close_price_sell = forward[:, col_dict['bids[0]']].mean()
        return sell_price - avg_close_price_sell, wallet
      else:
        return -fee/forward_ws, wallet
    else:  # sit
      if wallet == 0:
        return  0.0, wallet
      else:
        #return -fee/forward_ws, wallet
        return 0.0, wallet
