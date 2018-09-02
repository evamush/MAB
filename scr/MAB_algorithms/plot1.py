import numpy as np
import pandas as pd
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from Policy_contextfree import *
from policy_LinUCB import *
#from policy_LinUCB_decay import *
#from policy_HybridLinUCB import *
#from Policy_extra import *
#from numpy import linalg as LA

def runepoch(policy,views,epoch = 3):
    mean_reward=[]
    for i in range(epoch):
        x = policy.policy_evaluator(views)
        mean_reward.append(x)
        
    mean_reward = np.average(mean_reward,axis=0)
    return mean_reward

user_f = pd.read_csv("../../data/user_features/user/user_features_4.csv")
#change into binary
user_f.iloc[:,1:] = user_f.iloc[:,1:].astype(bool).astype(int)
#user_f.to_csv("../../data/user_features/user/user_features_3.csv",index=False,sep=',')
block_f = pd.read_csv('../../data/block_features/block_features.csv')
views = pd.read_csv('../../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)
print(len(blocks))


linucb = LinUCB(0.3,user_f, block_f,1)
r_0= runepoch(linucb,views)
#r_0 = linucb.policy_evaluator(views)

with open('ctr_order/linucb_4_no_arm_f.txt', 'w') as mc:
    for item in r_0:
        mc.write("%s\n" % item)

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
