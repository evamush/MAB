from stackplot import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import json
import time
from sklearn.decomposition import PCA

opt = 'https://www.petspyjamas.com/travel/nav/dog-friendly-cottages-hot-tub'
def runepoch(policy,views,epoch = 1):
    mean_reward=[]
    #cum_regret=[]
    #mean_regret=[]
    for i in range(epoch):
        x,_,_,df = policy.policy_evaluator(views)
        mean_reward.append(x)
        #cum_regret.append(y)
        #mean_regret.append(z)
    mean_reward = np.average(mean_reward,axis=0)
    #cum_regret = np.average(cum_regret,axis=0)
    #mean_regret = np.average(mean_regret,axis=0)
    return mean_reward,df

user_f = pd.read_csv("../../data/user_features/user/user_features_3.csv")
#user_f = pd.read_csv("../../data/user_features.csv")
block_f = pd.read_csv('../../data/block_features/block_features.csv')
views = pd.read_csv('../../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)
ts = Thompson_sampling(blocks)
rwd1,c = runepoch(ts,list(views['blockid']))

'''
linucb = LinUCBcheat(0.3,user_f, block_f,1)
rwd1,c = linucb.policy_evaluator()

linucb = LinUCB(0.3,user_f, block_f,1)
rwd1,c = runepoch(linucb,views)

ts = Thompson_sampling(blocks)
rwd1,c = runepoch(ts,list(views['blockid']))

ucb = UCB1(blocks,np.sqrt(2))
rwd1,c = runepoch(ucb,list(views['blockid']))
'''
data_perc = c.divide(c.sum(axis=1), axis=0)
l = [x.split('/')[-1] for x in blocks]
plt.figure(figsize=(12,8))
ax = plt.gca()
ax.set_title('100 % stacked area chart of LinUCB',  fontsize=18)
ax.set_xlabel('T', fontsize=14)
ax.set_ylabel('% of recommendations', fontsize=14)

plt.stackplot(range(0,len(c)),data_perc[blocks[0]],data_perc[blocks[1]],
    data_perc[blocks[2]],data_perc[blocks[3]],data_perc[blocks[4]],
    data_perc[blocks[5]],data_perc[blocks[6]],data_perc[blocks[7]],
    data_perc[blocks[8]],data_perc[blocks[9]],data_perc[blocks[10]],
    data_perc[blocks[11]],data_perc[blocks[12]],data_perc[blocks[13]],
    data_perc[blocks[14]],data_perc[blocks[15]],data_perc[blocks[16]],
    data_perc[blocks[17]],data_perc[blocks[18]],data_perc[blocks[19]])
'''
plt.stackplot(range(0,len(c)),data_perc[blocks[0]],data_perc[blocks[1]],
    data_perc[blocks[2]],data_perc[blocks[3]],data_perc[blocks[4]],
    data_perc[blocks[5]],data_perc[blocks[6]],data_perc[blocks[7]],
    data_perc[blocks[8]],data_perc[blocks[9]],data_perc[blocks[10]],
    data_perc[blocks[11]],data_perc[blocks[12]],data_perc[blocks[13]],
    data_perc[blocks[14]],data_perc[blocks[15]],data_perc[blocks[16]],
    data_perc[blocks[17]],data_perc[blocks[18]],data_perc[blocks[19]],
    labels=l)
'''
plt.legend(loc='upper right')
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.margins(0,0)
plt.show()
