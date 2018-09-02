import numpy as np
import pandas as pd
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Policy_contextfree import *
from policy_LinUCB_cheat import *
from Policy_extra import *

def runepoch(policy,epoch = 3):
    mean_reward=[]
    for i in range(epoch):
        x = policy.policy_evaluator()
        mean_reward.append(x)
    mean_reward = np.average(mean_reward,axis=0)
    return mean_reward

user_f = pd.read_csv("../../data/user_features/user/user_features_1.csv")
block_f = pd.read_csv('../../data/block_features/block_f.csv')
views = pd.read_csv('../../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)
print(len(blocks))

linucb = LinUCB(0.3,user_f, block_f,1)
r_0 = runepoch(linucb)
#r_0 = linucb.policy_evaluator()
#cheat with contextual features and the history

with open('ctr_order/cheatlinucb1_with_arm_f.txt', 'w') as mc:
    for item in r_0:
        mc.write("%s\n" % item)
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
ax = plt.gca()
ax.plot(r_0, label='MostClick')
#ax.plot(r_1, label='Clickrate')
ax.grid()
optimal=len(views[views['blockid']==opt])/len(views)
ax.axhline(y=optimal, color='k', linestyle=':', linewidth=2.5,label='optimal(hothub)')
ax.grid()
ax.legend(loc='lower right')
ax.set_title('20 blocks - sampling in random',  fontsize=18)
ax.set_xlabel('T', fontsize=14)
ax.set_ylabel('Cumulative average reward', fontsize=14)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()
'''
