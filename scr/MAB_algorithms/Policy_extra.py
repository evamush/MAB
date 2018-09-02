import random
import numpy as np
import pandas as pd
import time
import os
import json
import pymc

opt = 'https://www.petspyjamas.com/travel/nav/dog-friendly-cottages-hot-tub'
#opt = 'https://www.petspyjamas.com/travel/hub/dog-friendly-cotswolds/'

class MostClick():
    def __init__(self, arms):
        self.arms = arms
        self.k = len(arms)
        # Step count
        self.n = 0
        self.k_reward = np.zeros(self.k)#amount of reward we've gotten from each arms
        self.mean_reward = 0# Total mean reward
        self.a = 0
        return

    def select_arm(self):
        # generate a random number
        #max_p_t = np.max(self.k_reward)
        #max_idxs = np.argwhere(self.k_reward == max_p_t).flatten()
        #self.a = np.random.choice(max_idxs)
        self.a = np.argwhere(self.arms == opt)
 
        return

    def update(self, reward):
        self.n += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        self.k_reward[self.a] = self.k_reward[self.a] + reward 

        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.mean_reward = 0
        self.k_reward = np.zeros(self.k)
        return

    
    def policy_evaluator(self,views):
        # read existing logfile
        blocks = self.arms
        mean_reward = []
        cum_regret = []
        mean_regret = []
        regret = 0
        views_copy = views.copy()
        range_x = len(views)
        for i in range(range_x):
            href = views_copy.pop(random.choice(range(len(views_copy))))
            #if href == opt:
            self.select_arm()
            if self.arms[self.a] == href:
                reward = 1
            else:
                reward = 0
            
            regret += (1 - reward)
            cum_regret.append(regret)
            mean_regret.append(regret/(len(cum_regret)))
            self.update(reward)
            mean_reward.append(self.mean_reward)
        self.reset()
        return mean_reward[21:], cum_regret, mean_regret

class Clickrate():
    def __init__(self, arms):
        self.arms = arms
        self.k = len(arms)
        self.n = 0
        self.k_n = np.zeros(self.k)  # set count for each arm
        self.k_rate = np.ones(self.k)  # average amount of reward we've gotten from each arms
        self.mean_reward = 0
        self.a = 0
        return

    def select_arm(self):
        max_p_t = np.max(self.k_rate)
        max_idxs = np.argwhere(self.k_rate == max_p_t).flatten()
        self.a = np.random.choice(max_idxs)
        return 

    def update(self, reward):
        self.k_n[self.a] += 1
        self.n += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        
        rate = self.k_rate[self.a]
        new_rate = rate + (reward - rate) / (self.k_n[self.a] + 1)
        self.k_rate[self.a] = new_rate
        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.mean_reward = 0
        self.k_n = np.zeros(self.k)
        self.k_rate = np.zeros(self.k)
        return

    def policy_evaluator(self,views):
        # read existing logfile
        blocks = self.arms
        mean_reward = []
        cum_regret = []
        mean_regret = []
        regret = 0
        views_copy = views.copy()
        range_x = len(views)
        for i in range(range_x):
            href = views_copy.pop(random.choice(range(len(views_copy))))
            self.select_arm()
            print(self.arms[self.a])
            if self.arms[self.a] == href:
                reward = 1
            else:
                reward = 0
           
            regret += (1 - reward)
            cum_regret.append(regret)
            mean_regret.append(regret/(len(cum_regret)))
            self.update(reward)
            mean_reward.append(self.mean_reward)
        self.reset()
        return mean_reward[21:], cum_regret, mean_regret

    

    