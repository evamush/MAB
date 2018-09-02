import numpy as np
import pandas as pd
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Policy_contextfree import *
from policy_LinUCB import *
from policy_HybridLinUCB import *


opt = 'https://www.petspyjamas.com/travel/nav/dog-friendly-cottages-hot-tub'
#opt = 'https://www.petspyjamas.com/travel/hub/dog-friendly-cotswolds/'
def runepoch(policy,views,epoch = 3):
    mean_reward=[]
    for i in range(epoch):
        x = policy.policy_evaluator(views)
        mean_reward.append(x)
    mean_reward = np.average(mean_reward,axis=0)
    return mean_reward

block_f = pd.read_csv('../../data/block_features/block_features.csv')
views = pd.read_csv('../../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)
print(len(blocks))

e_greedy = EpsilonGreedy(0.1,blocks)
r_0 = runepoch(e_greedy,list(views['blockid']))
with open('ctr_order/E_greedy_01.txt', 'w') as E_greedy_01:
    for item in r_0:
        E_greedy_01.write("%s\n" % item)

e_greedy = EpsilonGreedy(1,blocks)
r_11 = runepoch(e_greedy,list(views['blockid']))
with open('ctr_order/random.txt', 'w') as E_greedy_1:
    for item in r_11:
        E_greedy_1.write("%s\n" % item)

ucb1 = UCB1(blocks,np.sqrt(2))
r_1 = runepoch(ucb1,list(views['blockid']))
with open('ctr_order/ucb1.txt', 'w') as ucb1:
    for item in r_1:
        ucb1.write("%s\n" % item)

ts = Thompson_sampling(blocks)
r_2 = runepoch(ts,list(views['blockid']))
with open('ctr_order/ts.txt', 'w') as ts:
    for item in r_2:
        ts.write("%s\n" % item)


import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
ax = plt.gca()
ax.plot(r_0, label='E_greedy 0.1')
ax.plot(r_11, label='random')#random
ax.plot(r_1, label='UCB1')
ax.plot(r_2, label='Thompson sampling')


optimal=len(views[views['blockid']==opt])/len(views)
ax.axhline(y=optimal, color='k', linestyle=':', linewidth=2.5,label='Cottages with Hot Tubs')
ax.grid()
ax.legend(loc='lower right')
ax.set_xlabel('T', fontsize=14)
ax.set_ylabel('CTR', fontsize=14)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()
