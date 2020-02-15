
class Trader:
  def __init__(self, enable_fee = False, fee_rate=0.0, record=False):
    self.pos = {}
    self.avgcost = {}
    self.pnl = {}
    self.trade_count = {}
    self.fee_rate = fee_rate
    self.record = record
    if record  == True:
      self.f = open('traders_record.txt', 'w')

  def __def__(self):
    if self.record:
      self.f.close()

  def RegisterOneTrade(self, ticker, size, price):
    assert price > 0
    assert isinstance(size, int)
    assert isinstance(ticker, str)
    if ticker not in self.pos:
      self.pos[ticker] = size
      self.avgcost[ticker] = price
      self.pnl[ticker] = 0.0
      self.trade_count[ticker] = 1.0
      return
    if self.pos[ticker] * size < 0:  # close position
      if abs(size) > self.pos[ticker]:  # over close
        self.pnl[ticker] += (self.avgcost[ticker] - price) * -self.pos[ticker]
        self.avgcost[ticker] = price
      else: # normal close
        self.pnl[ticker] += (self.avgcost[ticker] - price) * size
    else:  # open position
      self.avgcost[ticker] += (price-self.avgcost[ticker])*size/(self.pos[ticker]+size)
    self.pos[ticker] += size
    self.trade_count[ticker] += 1
    trade_record = "Trade %s %d@%f" %(ticker, size, price)
    if self.record:
      self.f.write(trade_record+'\n')

  def Summary(self):
    print('================================================================================================================')
    for t in self.pos:
      print(t)
      print(self.trade_count[t])
      print(self.pnl[t])
      print(self.pos[t])
      print(self.avgcost[t])
      print(self.fee_rate)
      print('for %s trade_count=%d, pnl=%.2f, left_pos=%.2f, avgcost=%.2f, for feerate %f, rough_fee is %.2f' %(t, self.trade_count[t], self.pnl[t], self.pos[t], self.avgcost[t], self.fee_rate, abs(self.fee_rate*self.trade_count[t]*self.avgcost[t])))
    print('===============================================================================================================')
