from RLTrading import *

rltrader = RLTrading()

if __name__ == '__main__':
  rltrader.Train(epochs=1)
  rltrader.Test()
