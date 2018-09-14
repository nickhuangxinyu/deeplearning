# -*- coding=utf-8 -*-
from hmmlearn import hmm
import numpy as np

states = ["Rainy", "Sunny"]##隐藏状态
n_states = len(states)##隐藏状态长度

observations = ["walk", "shop", "clean"]##可观察的状态
n_observations = len(observations)##可观察序列的长度

start_probability = np.array([0.6, 0.4])##开始转移概率，即开始是Rainy和Sunny的概率
##隐藏间天气转移混淆矩阵，即Rainy和Sunny之间的转换关系，例如[0,0]表示今天Rainy，明天Rainy的概率
transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])
##隐藏状态天气和可视行为混淆矩阵，例如[0,0]表示今天Rainy，walk行为的概率为0.1
emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

#构建了一个MultinomialHMM模型，这模型包括开始的转移概率，隐藏间天气转换混淆矩阵（transmat），隐藏状态天气和可视行为混淆矩阵emissionprob，对模型参数初始化
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_= start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#给出一个可见序列
bob_Actions = np.array([[2, 0, 1, 1, 2, 0]]).T

# 解决问题1,解码问题,已知模型参数和X，估计最可能的Z； 维特比算法 
logprob, weathers = model.decode(bob_Actions, algorithm="viterbi")
print "weathers:", ", ".join(map(lambda x: states[x], weathers))
