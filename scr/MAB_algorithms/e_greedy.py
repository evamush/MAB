from Policy_contextfree import *
from policy_LinUCB import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import json
import time

block_f = pd.read_csv('../../data/block_features/block_features.csv')
views = pd.read_csv('../../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)
def runepoch(policy,views,epoch = 3):
    mean_reward=[]
    for i in range(epoch):
        x = policy.policy_evaluator(views)
        mean_reward.append(x)
    mean_reward = np.average(mean_reward,axis=0)
    return mean_reward

e_greedy = EpsilonGreedy(0, blocks)
rwd0= runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.1, blocks)
rwd1=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.2, blocks)
rwd2=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.3, blocks)
rwd3=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.4, blocks)
rwd4=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.5, blocks)
rwd5=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.6, blocks)
rwd6=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.7, blocks)
rwd7=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.8, blocks)
rwd8=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(0.9, blocks)
rwd9=runepoch(e_greedy,list(views['blockid']))
e_greedy = EpsilonGreedy(1, blocks)
rwd10=runepoch(e_greedy,list(views['blockid']))

plt.figure(figsize=(12,8))
ax = plt.gca()


ax.plot(rwd0, label='0.0 e-greedy')
ax.plot(rwd1, label='0.1 e-greedy')
ax.plot(rwd2, label='0.2 e-greedy')
ax.plot(rwd3, label='0.3 e-greedy')
ax.plot(rwd5, label='0.5 e-greedy')
ax.plot(rwd6, label='0.6 e-greedy')
ax.plot(rwd7, label='0.7 e-greedy')
ax.plot(rwd8, label='0.8 e-greedy')
ax.plot(rwd9, label='0.9 e-greedy')
ax.plot(rwd10, label='1.0 e-greedy')


ax.legend()
plt.xlabel('T')
plt.ylabel('CTR')
plt.show()

