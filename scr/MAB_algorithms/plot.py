egreedy1 = []
with open('ctr_old/E_greedy_1_order.txt') as f:
    for line in f:
        egreedy1.append(float(line.split('\n')[0]))

egreedy01 = []
with open('ctr_old/E_greedy_01_order.txt') as f:
    for line in f:
        egreedy01.append(float(line.split('\n')[0]))

TS = []
with open('ctr_old/tsorder.txt') as f:
    for line in f:
        TS.append(float(line.split('\n')[0]))

UCB1 = []
with open('ctr_old/ucb1order.txt') as f:
    for line in f:
        UCB1.append(float(line.split('\n')[0]))

linucb = []
with open('ctr_old/linucb_order.txt') as f:
    for line in f:
        linucb.append(float(line.split('\n')[0]))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(12,8))
ax = plt.gca()
ax.plot(egreedy01,  label='E_greedy_0.1')
ax.plot(egreedy1,label='egreedy_1.0')
ax.plot(UCB1, label='UCB1')
ax.plot(TS,label='TS')
ax.plot(linucb,label='LinUCB')
opt = 'https://www.petspyjamas.com/travel/nav/dog-friendly-cottages-hot-tub'
views = pd.read_csv('../../data/views.csv')
optimal=len(views[views['blockid']==opt])/len(views)
ax.axhline(y=optimal, color='k', linestyle=':', linewidth=2.5,label='optimal(hothub)')
ax.grid()
ax.legend(loc='lower right')
ax.set_title('20 blocks - sampling by timestep',  fontsize=18)
ax.set_xlabel('T', fontsize=14)
ax.set_ylabel('Cumulative average reward', fontsize=14)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()