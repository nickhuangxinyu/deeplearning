from abc import ABCMeta, abstractmethod

class rl(metaclass=ABCMeta):
  def __init__(self):
    pass
  @abstractmethod
  def init(self):
    pass
  @abstractmethod
  def step(self, action, current):
    pass
  @abstractmethod
  def update(self):
    pass
  @abstractmethod
  def store_memory(self):
    pass
  @abstractmethod
  def choose_action(self):
    pass
