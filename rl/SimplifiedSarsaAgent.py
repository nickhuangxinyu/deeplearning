#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# A simple sarsa Agent for discrete ovservation and action states

# Author: Qiang Ye
# Date: July 22, 2017
# License: MIT

from random import random
from gym import Env
import gym
import sys
sys.path.append('..')
from gridworld import *
from collections import Counter
import pandas as pd

class SarsaAgent(object):
    def __init__(self, env:Env):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env
        self.Q = {}  # {s0:[,,,,,,],s1:[,,,,,]} 数组内元素个数为行为空间大小
        self._initAgent()
        self.state = None

    def _get_state_name(self, state):   # 得到状态对应的字符串作为以字典存储的价值函数
        return str(state)               # 的键值，应针对不同的状态值单独设计，避免重复
                                        # 这里仅针对格子世界
    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 1.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):
        # 　cann't find the state
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)
    
    def _get_Q(self, s, a):
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self, s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def _initAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized = False)
    
    # using simple decaying epsilon greedy exploration
    def _curPolicy(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / (episode_num+1)
        #epsilon = 0.1
        Q_s = self.Q[int(s)]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:  
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
            '''
            max_v = Q_s[max(Q_s)]
            ml = []
            for i in Q_s:
              if Q_s[i] == max_v:
                ml.append(i)
            interval = 1.0/len(ml)
            rand = random()
            ith = int(rand/interval)
            action = ith
            '''
        return action

    # Agent依据当前策略和状态决定下一步的动作
    def performPolicy(self, s, episode_num, use_epsilon=True):
        return self._curPolicy(s, episode_num, use_epsilon)

    def act(self, a):
        return self.env.step(a)

    def ptq(self):
        print(self.Q)

    def putcsv(self, filename = 'test.csv'):
      l = np.array([[0.0] * 70] * 4)
      for i in self.Q.items():
        st = i[0]
        mp = i[1]
        for j in mp.items():
          at = j[0]
          value = j[1]
          l[int(at), int(st)] = value
      pd.DataFrame(l).to_csv(filename)
      return l.transpose()

    def BestD(self, filename='test.csv'):
      l = self.putcsv(filename)
      init_state = int(self._get_state_name(self.state))
      s0 = init_state
      self.env.state = s0
      # print(s0)
      route = []
      while s0 != 37:
        at = np.argmax(l[s0])
        route.append(at)
        print(l[s0])
        print("for state %d, take action %d" %(s0, at))
        s0, temp_r, done, info = self.act(at)
        s0 = int(s0)
        print('new state %d' %(s0))
      print(route)
      print('route length is ', len(route))
        
 
    def mc_learning(self, gamma, alpha, max_episode_num):
        total_time, time_in_episode, num_episode = 0, 0, 0
        sa_count = {}
        while num_episode < max_episode_num:
            sa = []
            self.state = self.env.reset()
            r = []
            self.env.render()
            done = False
            num_step = 0
            s0 = int(self._get_state_name(self.state))
            #s0 = int(random() *31)
            #print('init state is ', s0)
            while not done:
              self._assert_state_in_Q(s0, randomized = False)
              if s0 not in sa_count:
                sa_count[s0] = {i: 0 for i in self.Q[s0]}
              temp_a = self.performPolicy(s0, num_episode, use_epsilon = True)
              sa.append((s0,temp_a))
              s0, temp_r, done, info = self.act(temp_a)
              self.env.render()
              r.append(temp_r)
              num_step += 1
            seq_length = len(r)
            for i in range(len(sa)):
              gt = 0.0
              st = sa[i][0]
              at = sa[i][1]
              sa_count[st][at] += 1
              for j in range(i,len(sa)):
                gt += pow(gamma, j-i+1) * r[j]
              gt += -1
              self.Q[st][at] += (gt-self.Q[st][at]) / sa_count[st][at]
            num_episode += 1
            print('%dth learning cost %d step, gt is %lf' %(num_episode, num_step, gt))
            print(r)
            print(sa)
            #self.putcsv(filename='mc.csv')
            #string = input()
            #if (string == 'q'):
              #self.putcsv(filename='mc.csv')
        '''
        l = np.array([[0.0] * 70] * 4)
        for i in self.Q.items():
          st = i[0]
          mp = i[1]
          for j in mp.items():
            at = j[0]
            value = j[1]
            l[int(at), int(st)] = value
        pd.DataFrame(l).to_csv('mc.txt')
        '''
        #self.BestD(filename='mc.csv')
            
    # sarsa learning
    def learning(self, gamma, alpha, max_episode_num):
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        total_time, time_in_episode, num_episode = 0, 0, 0

        while num_episode < max_episode_num:
            self.state = self.env.reset()
            s0 = int(self._get_state_name(self.state))
            self._assert_state_in_Q(s0, randomized = False)
            self.env.render()
            a0 = self.performPolicy(s0, num_episode, use_epsilon = True)
            
            time_in_episode = 0
            is_done = False
            while not is_done:
                # a0 = self.performPolicy(s0, num_episode)
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = int(self._get_state_name(s1))
                self._assert_state_in_Q(s1, randomized = False)
                # 在下行代码中添加参数use_epsilon = False即变成Q学习算法
                # real action
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                # update action
                a2 = self.performPolicy(s1, num_episode, use_epsilon=False)
                old_q = self._get_Q(s0, a0)
                # 
                q_prime = self._get_Q(s1, a2)
                td_target = r1 + gamma * q_prime
                #alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)
                
                if num_episode == max_episode_num:
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".\
                        format(time_in_episode, s0, a0, s1))

                s0, a0 = s1, a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps.".format(
                num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1
        '''
        l = np.array([[0.0] * 70] * 4)
        for i in self.Q.items():
          st = i[0]
          mp = i[1]
          for j in mp.items():
            at = j[0]
            value = j[1]
            l[int(at), int(st)] = value
        pd.DataFrame(l).to_csv('sarsa.txt')
        '''
        self.BestD(filename='sarsa.txt')
        return   



def main():

    #env = gym.make("WindyGridWorld-v0")
    #directory = "/home/qiang/workspace/reinforce/python/monitor"
    
    #env = gym.wrappers.Monitor(env, directory, force=True)
    #env = WindyGridWorld()
    env = SimpleGridWorld()
    agent = SarsaAgent(env)
    env.reset()
    print("Learning...")  
    '''
    agent.mc_learning(gamma=0.9, 
                   alpha=0.1, 
                   max_episode_num=300)
    '''
    agent.learning(gamma=0.9, 
                   alpha=0.1, 
                   max_episode_num=300)
    env.close()


if __name__ == "__main__":
    main()
