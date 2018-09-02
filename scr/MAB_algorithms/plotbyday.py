import numpy as np
import pandas as pd
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Policy_contextfree import *
from policy_LinUCB import *

user_f = pd.read_csv("../../data/user_features/user/user_features_3.csv")
block_f = pd.read_csv('../../data/block_features/block_features.csv')
views = pd.read_csv('../../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)
print(len(blocks))
'''
linucb = LinUCB(0.3,user_f, block_f,1)
r1=[]
ucb1 = UCB1(blocks,np.sqrt(2))
e_greedy = EpsilonGreedy(0.1,blocks)
randome = EpsilonGreedy(1,blocks)
ts = Thompson_sampling(blocks)

r2=[]
r3=[]
r4=[]
r5=[]
file = ['06/22','06/23','06/24','06/25','06/26','06/27','06/28',
        '06/29','06/30','07/01','07/02','07/03','07/04','07/05',
        '07/06','07/07','07/08','07/09','07/10','07/11','07/12',
        '07/13','07/14','07/15','07/16','07/17','07/18','07/19',
        '07/20','07/21','07/22','07/23','07/24','07/25','07/26',
        '07/27','07/28','07/29','07/30','07/31','08/01','08/02',
        '08/03','08/04','08/05','08/06','08/07','08/08','08/09',
        '08/10','08/11','08/12','08/13']
days = 0
for i in file:
    days += 1
    rewards_linucb = []
    rewards_ucb1 = []
    rewards_e = []
    rewards_random = []
    rewards_ts = []
    logfile = "../../impression-2018/"+ i 
    g = os.walk(logfile)
    for path, d, filelist in sorted(g):
        for filename in sorted(filelist):
            file = open(os.path.join(path, filename))
            for line in file:
                l = json.loads(line)
                href = l.setdefault('meta').setdefault('href')
                if href in blocks:
                    timestamp = l.setdefault('meta').setdefault('date').setdefault('date').split('.')[0].split('T')
                    userid = int(time.mktime(time.strptime(timestamp[0] + ' ' + timestamp[1], "%Y-%m-%d %H:%M:%S")))
                
                    user_feature = user_f[user_f.userid == userid].iloc[:,1:].values[0] 

                    r_linucb = linucb.recommend(user_feature)
                    if href in r_linucb:
                        reward_linucb = 1
                    else:
                        reward_linucb = 0
                     
                    ucb1.select_arm()
                    if ucb1.arms[ucb1.a] == href :
                        reward_ucb1 = 1
                    else:
                        reward_ucb1 = 0
                        
                    e_greedy.select_arm()
                    if e_greedy.arms[e_greedy.a] == href :
                        reward_e = 1
                    else:
                        reward_e = 0 
                        
                    randome.select_arm()
                    if randome.arms[randome.a] == href:
                        reward_random = 1
                    else:
                        reward_random = 0
                    
                    ts.select_arm()
                    if ts.arms[ts.a] == href:
                        reward_ts = 1
                    else:
                        reward_ts = 0
                    
                    linucb.update(reward_linucb)
                    
                    ucb1.update(reward_ucb1)
                    e_greedy.update(reward_e)
                    randome.update(reward_random)
                    ts.update(reward_ts)
                    rewards_linucb.append(reward_linucb)
                    rewards_ucb1.append(reward_ucb1)
                    rewards_e.append(reward_e)
                    rewards_random.append(reward_random)
                    rewards_ts.append(reward_ts)
    if days % 3 == 0:
        #linucb.reset()     
        r1.append(np.average(np.array(rewards_linucb)))
        #ucb1.reset()
        r2.append(np.average(np.array(rewards_ucb1)))
        #e_greedy.reset()
        r3.append(np.average(np.array(rewards_e)))
        #randome.reset()
        r4.append(np.average(np.array(rewards_random)))
        #ts.reset()
        r5.append(np.average(np.array(rewards_ts)))
'''       
from policy_LinUCB_cheat import *
cheatlinucb = LinUCB(0.3,user_f, block_f,1)

r6=[]

file = ['06/22','06/23','06/24','06/25','06/26','06/27','06/28',
        '06/29','06/30','07/01','07/02','07/03','07/04','07/05',
        '07/06','07/07','07/08','07/09','07/10','07/11','07/12',
        '07/13','07/14','07/15','07/16','07/17','07/18','07/19',
        '07/20','07/21','07/22','07/23','07/24','07/25','07/26',
        '07/27','07/28','07/29','07/30','07/31','08/01','08/02',
        '08/03','08/04','08/05','08/06','08/07','08/08','08/09',
        '08/10','08/11','08/12','08/13']
days = 0
new = dict.fromkeys(blocks, 0)
save_dict('views.npy',{})
views = load_dict('views.npy')
for i in file:
    days += 1
    cheatlinucb.mean_reward = 0
    cheatlinucb.n = 0
    logfile = "../../impression-2018/"+ i 
    g = os.walk(logfile)
    for path, d, filelist in sorted(g):
        for filename in sorted(filelist):
            file = open(os.path.join(path, filename))
            for line in file:
                l = json.loads(line)
                href = l.setdefault('meta').setdefault('href')
                if href in blocks:
                    userid = l.setdefault('MediaGammaImpression', None)
                    timestamp = l.setdefault('meta').setdefault('date').setdefault('date').split('.')[0].split('T')
                    timestamp = int(time.mktime(time.strptime(timestamp[0] + ' ' + timestamp[1], "%Y-%m-%d %H:%M:%S")))

                    if views == {}:
                        user_feature = np.zeros(len(blocks))
                        views[userid] = new.copy()
                    else:
                        if userid in views.keys():
                            user_feature = np.array(list(views[userid].values()))
                        else:
                            views[userid] = new.copy()
                            user_feature = np.zeros(len(blocks))

                    user_feature = user_f[user_f.userid == timestamp].iloc[:,1:].values[0] 
                    recommends = cheatlinucb.recommend(user_feature)
                    if href in recommends:
                        reward = 1
                        views[userid][cheatlinucb.a_max] = views[userid][cheatlinucb.a_max] + 1
                        
                        cheatlinucb.n += 1
                        cheatlinucb.mean_reward = cheatlinucb.mean_reward + (1 - cheatlinucb.mean_reward) / cheatlinucb.n
                        cheatlinucb.update(reward)
                    else:
                        x=np.array(list(views[userid].values()))
                        user_pos_rat= np.where(x!=0)[0]
                        num_known_ratings = len(user_pos_rat)+1
                        
                        likability = 0
                        a = pd.DataFrame.from_dict(views)
                        if len(a)>2000:
                            a = a.iloc[:,:2000].T
                        else:
                            a=a.T
                        for item in user_pos_rat:
                            likability += list(a.corrwith(a[cheatlinucb.a_max]).fillna(0))[item]
                        likability /= num_known_ratings
                        #print(likability)
                        binomial_reward_probability = likability
                        if binomial_reward_probability <= 0:
                        #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
                            binomial_reward_probability = 0 # this could be replaced by small probability
                        approx_rating = np.random.binomial(n=1, p=binomial_reward_probability)  # Bernoulli coin toss
                        if approx_rating == 1:
                            print('cheat')
                            reward = 1
                            views[userid][cheatlinucb.a_max] = views[userid][cheatlinucb.a_max] + 1
                        else:
                            reward = 0
                        cheatlinucb.n += 1
                        cheatlinucb.mean_reward = cheatlinucb.mean_reward + (0 - cheatlinucb.mean_reward) / cheatlinucb.n
                        cheatlinucb.update(reward)
                    #avg_reward = np.average(np.array(rewards))
                    #views[userid][href] = 1
                    #print(cheatlinucb.n,cheatlinucb.mean_reward)
                    #avg_rewards.append(avg_reward)
    avg_rewards=cheatlinucb.mean_reward
        
    if days % 5 == 0:
        #cheatlinucb.reset()     
        r6.append(avg_rewards)
        print(avg_rewards)
'''        
with open('ctr_day/3/LinUCB2.txt', 'w') as linucb:
    for item in r1:
        linucb.write("%s\n" % item)

with open('ctr_day/3/E_greedy_01.txt', 'w') as E_greedy_01:
    for item in r3:
        E_greedy_01.write("%s\n" % item)

with open('ctr_day/3/random.txt', 'w') as E_greedy_1:
    for item in r4:
        E_greedy_1.write("%s\n" % item)


with open('ctr_day/3/ucb1.txt', 'w') as ucb1:
    for item in r2:
        ucb1.write("%s\n" % item)

with open('ctr_day/3/ts.txt', 'w') as ts:
    for item in r5:
        ts.write("%s\n" % item)
'''
with open('ctr_day/3/clinucb5.txt', 'w') as clinucb:
    for item in r6:
        clinucb.write("%s\n" % item)
plt.figure(figsize=(12,8))
ax = plt.gca()
ax.plot(r6, label='Cheatlinucb')
plt.show()
'''
plt.figure(figsize=(12,8))
ax = plt.gca()
ax.plot(r3, label='E_greedy 0.1')
ax.plot(r4, label='random')#random
ax.plot(r2, label='UCB1')
ax.plot(r5, label='Thompson sampling')
ax.plot(r1, label='LinUCB')
ax.plot(r6, label='Cheatlinucb')
ax.grid()
ax.legend(loc='lower right')
ax.set_title('20 blocks - sampling in random',  fontsize=18)
ax.set_xlabel('T', fontsize=14)
ax.set_ylabel('Cumulative average reward', fontsize=14)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()
'''
