import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(100,20)*100
w = np.random.randn(20,1)*10
noise = np.random.randn(100,1)*2
y = np.dot(x, w) + noise

def nn(x, w):
    return np.dot(x, w)

def delta(x, w, y):
    return nn(x, w)-y

def loss(x, w, y):
    return (delta(x,w,y)**2).sum()

def gradient(x, w, y):
    return 2*np.dot(np.transpose(x), np.dot(x,w)-y)

init_w = w

w = np.zeros(20).reshape(20,1)

l = 0.0
i = 0
alpha = 0.0001
while abs(loss(x, w, y) - l) > 0.1:
   l = loss(x,w,y)
   w = w - alpha*gradient(x,w,y)
   i = i + 1  
   print(i, "th update: loss is ", l)
