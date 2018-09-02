import random
import numpy as np
import pandas as pd
import time
import os
import json
import pymc
from sklearn.decomposition import PCA

opt = 'https://www.petspyjamas.com/travel/nav/dog-friendly-cottages-hot-tub'

class EpsilonGreedy():
    def __init__(self, epsilon, arms):
        self.arms = arms
        self.k = len(arms)
        # Step count
        self.n = 0
        self.k_n = np.zeros(self.k) # set count for each arm
        self.k_reward = np.zeros(self.k)# average amount of reward we've gotten from each arms
        self.mean_reward = 0# Total mean reward
        self.epsilon = epsilon # probability of explore
        self.a = 0
        return

    def select_arm(self):
        # generate a random number
        if random.random() > self.epsilon:
            self.a = np.argmax(self.k_reward)
        else:
            self.a = np.random.choice(self.k)
        return

    def update(self, reward):
        self.k_n[self.a] += 1
        self.n += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        self.k_reward[self.a] = self.k_reward[self.a] + (reward - self.k_reward[self.a])\
                                /self.k_n[self.a]
        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.k_reward = np.zeros(self.k)
        return

    def policy_evaluator(self,views):
        # read existing logfile
        blocks = self.arms
        count_ = dict.fromkeys(blocks, 0)
        c = []
        mean_reward = []
        cum_regret = []
        mean_regret = []
        regret = 0
        views_copy = views.copy()
        range_x = len(views)
        for i in range(range_x):
            #href = views_copy.pop(random.choice(range(len(views_copy))))
            href = views_copy.pop(0)
            self.select_arm()
            count_[self.arms[self.a]] += 1
            if self.arms[self.a] == href:
                reward = 1
            else:
                reward = 0
            mean_reward.append(self.mean_reward)
            if href == opt:
                regret += (1 - reward)
            else:
                regret += (0 - reward)
            cum_regret.append(regret)
            mean_regret.append(regret/(len(cum_regret)))
            self.update(reward)
            c.append(list(count_.values()))
        df = pd.DataFrame(c, columns=blocks)
        self.reset()
        return mean_reward, cum_regret, mean_regret,df

class UCB1():
    def __init__(self, arms, c):
        self.arms = arms
        self.k = len(arms)
        self.c = c
        self.n = 0
        self.k_n = np.zeros(self.k)  # set count for each arm
        self.k_reward = np.zeros(self.k)  # average amount of reward we've gotten from each arms
        self.mean_reward = 0
        self.a = 0
        return

    def select_arm(self):
        # generate a random number
        for arm in range(self.k):
            if self.k_n[arm] == 0:
                self.a = arm
                return 

        ucb_values = [0.0 for arm in range(self.k)]
        for arm in range(self.k):
            ucb_values[arm] = self.k_reward[arm] + self.c * (np.sqrt(np.log(self.n)) / float(self.k_n[arm]))
        #self.a = np.argmax(ucb_values)
        max_p_t = np.max(ucb_values)
        max_idxs = np.argwhere(ucb_values == max_p_t).flatten()
        self.a = np.random.choice(max_idxs)
        return 

    def update(self, reward):
        self.k_n[self.a] += 1
        self.n += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        self.k_reward[self.a] = self.k_reward[self.a] + (reward - self.k_reward[self.a])\
                                /self.k_n[self.a]
        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.mean_reward = 0
        self.k_n = np.zeros(self.k)
        self.k_reward = np.zeros(self.k)
        return

    def policy_evaluator(self,views):
        # read existing logfile
        blocks = self.arms
        count_ = dict.fromkeys(blocks, 0)
        c = []
        mean_reward = []
        cum_regret = []
        mean_regret = []
        regret = 0
        views_copy = views.copy()
        range_x = len(views)
        for i in range(range_x):
            #href = views_copy.pop(random.choice(range(len(views_copy))))
            href = views_copy.pop(0)
            self.select_arm()
            count_[self.arms[self.a]] += 1
            if self.arms[self.a] == href:
                reward = 1
            else:
                reward = 0
            self.update(reward)
            mean_reward.append(self.mean_reward)
            if href == opt:
                regret += (1 - reward)
            else:
                regret += (0 - reward)
            cum_regret.append(regret)
            mean_regret.append(regret/(len(cum_regret)))
            
            c.append(list(count_.values()))
        df = pd.DataFrame(c, columns=blocks)
        self.reset()
        return mean_reward, cum_regret, mean_regret,df

class Thompson_sampling():
    def __init__(self, arms):
        self.arms = arms
        self.k = len(arms)
        self.n = 0
        self.k_reward = np.zeros(self.k)  # set count for each arm
        self.k_n = np.zeros(self.k)  # average amount of reward we've gotten from each arms
        self.wins = np.zeros(self.k)
        self.mean_reward = 0
        self.a = 0
        return

    def select_arm(self):
        #self.a = np.argmax(pymc.rbeta(1 + self.wins, 1 + self.k_n - self.wins))
        ts_values = pymc.rbeta(1 + self.wins, 1 + self.k_n - self.wins)
        max_p_t = np.max(ts_values)
        max_idxs = np.argwhere(ts_values == max_p_t).flatten()
        self.a = np.random.choice(max_idxs)
        return

    def update(self, reward):
        self.n += 1
        self.k_n[self.a] += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        self.k_reward[self.a] = self.k_reward[self.a] + (reward - self.k_reward[self.a])\
                                /self.k_n[self.a]
        self.wins[self.a] = self.k_reward[self.a] * self.k_n[self.a]
        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.mean_reward = 0
        self.k_n = np.zeros(self.k)
        self.k_reward = np.zeros(self.k)
        self.wins = np.zeros(self.k)  # set count for each arm
        return

    def policy_evaluator(self,views):
        # read existing logfile
        blocks = self.arms
        count_ = dict.fromkeys(blocks, 0)
        c = []
        mean_reward = []
        cum_regret = []
        mean_regret = []
        regret = 0
        views_copy = views.copy()
        range_x = len(views)
        for i in range(range_x):
            #href = views_copy.pop(random.choice(range(len(views_copy))))
            href = views_copy.pop(0)
            self.select_arm()
            count_[self.arms[self.a]] += 1
            if self.arms[self.a] == href:
                reward = 1
            else:
                reward = 0
            self.update(reward)
            mean_reward.append(self.mean_reward)
            if href == opt:
                regret += (1 - reward)
            else:
                regret += (0 - reward)
            cum_regret.append(regret)
            mean_regret.append(regret/(len(cum_regret)))

            c.append(list(count_.values()))
        df = pd.DataFrame(c, columns=blocks)
        self.reset()
        return mean_reward, cum_regret, mean_regret,df

def save_dict(filename,dictionary):
    np.save(filename,dictionary)

def load_dict(filename):
    dict = np.load(filename).item()
    return dict

def update_dict(new,old):
    #used to update the user views history
    #new with all items set to zero
    new.update(old)
    return new

class LinUCB:
    def __init__(self,alpha,user_f, block_f,n_recom):
        self.alpha = alpha  # explore depth
        self.block_f = block_f
        self.user_f = user_f
        self.n_recom = n_recom
        #self.d = self.user_f.shape[1] + self.block_f.shape[1] - 2 # dimension of user features
        self.d = self.user_f.shape[1] - 1
        #self.d = 6
        self.Aa = {}
        # collection of matrix to compute disjoint part of each block a d*d
        #self.AaI = {}  inverse of all Aa matrix
        self.ba = {}
        # collection of vectors to compute disjoint part, d*1
        self.a_max = 0
        self.x = None
        #self.theta = {}   linear parameter
        self.blocks = list(block_f.blockid)
        for key in self.blocks:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))

    def reset(self):
        for key in self.blocks:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
        return    
    def update(self, r):
        self.Aa[self.a_max] = self.Aa[self.a_max] + self.x.dot(self.x.T)
        self.ba[self.a_max] = self.ba[self.a_max] + r * self.x

    def recommend(self, user_feature):
        blocks = self.blocks
        n_recom = self.n_recom
        Aa = self.Aa
        ba = self.ba
        user_features = np.tile(user_feature, (len(blocks), 1))
        #block_features = np.asmatrix(self.block_f.iloc[:, 1:].values)
        #arm_features = np.concatenate((user_features, block_features), axis=1)
        
        arm_features = user_features
        
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        i = 0
        for block in blocks:
            xa = arm_features[i].reshape(arm_features.shape[1], 1)
            AaI_temp = np.linalg.inv(Aa[block])
            theta_temp = AaI_temp.dot(ba[block])
            p_t[i] = theta_temp.T.dot(xa) + self.alpha * np.sqrt(xa.T.dot(AaI_temp).dot(xa))
            i += 1

        max_p_t = np.max(p_t)
        if max_p_t <= 0:
            print("max p_t={}, p_t={}".format(max_p_t, p_t))
        # I want to randomly break ties, np.argmax return the first occurence of maximum.
        # So I will get all occurences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to user t
        a_max = self.blocks[a_t]
        
        r_list = []
        for i in (np.argpartition(p_t.flatten(), -n_recom)[-n_recom:]):
            r_list.append(blocks[i])
        self.a_max = a_max
        self.x = arm_features[a_t].reshape(arm_features.shape[1], 1)
      
        return r_list

  
    def policy_evaluator(self, views_f):
        # read existing logfile
        blocks = self.blocks
        user_f = self.user_f
        rewards = []
        expected_rewards = []
        count_ = dict.fromkeys(blocks, 0)
        c = []
        avg_rewards = []
        avg_expected_rewards = []
        regret = 0
        cum_regret = []
        mean_regret = []
        range_x = len(views_f)
        u = list(views_f.userid)
        h = list(views_f.blockid)
        print('ordered')
        #print('random')
        for i in range(range_x):
            #idx = random.choice(range(len(u)))
            idx = 0
            userid = u.pop(idx)
            href = h.pop(idx)
            user_feature = user_f[user_f.userid == userid].iloc[:,1:].values[0] 
            
            recommends = self.recommend(user_feature)   
            count_[self.a_max] += 1
            if href in recommends:
                reward = 1
            else:
                reward = 0
            self.update(reward)
            c.append(list(count_.values())) 
            rewards.append(reward)
            avg_reward = np.average(np.array(rewards))
            avg_rewards.append(avg_reward)
            if href == opt:
                regret += (1 - reward)
            else:
                regret += (0 - reward)
            cum_regret.append(regret)
            mean_regret.append(regret/len(cum_regret))
        df = pd.DataFrame(c, columns=blocks)   
        return avg_rewards, cum_regret, mean_regret,df

class LinUCBcheat:
    def __init__(self,alpha,user_f, block_f,n_recom):
        self.alpha = alpha  # explore depth
        self.block_f = block_f
        self.user_f = user_f
        self.n_recom = n_recom
        self.mean_reward = 0
        #self.d = self.user_f.shape[1] + self.block_f.shape[1] - 2 # dimension of user features
        self.d = self.user_f.shape[1] - 1
        #self.d = 6
        self.Aa = {}
        # collection of matrix to compute disjoint part of each block a d*d
        #self.AaI = {}  inverse of all Aa matrix
        self.ba = {}
        # collection of vectors to compute disjoint part, d*1
        self.a_max = 0
        self.x = None
        self.n = 0
        #self.theta = {}   linear parameter
        self.blocks = list(block_f.blockid)
        for key in self.blocks:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))

    def reset(self):
        self.n=0
        self.mean_reward=0
        for key in self.blocks:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
        return    

    def update(self, r):
        self.Aa[self.a_max] = self.Aa[self.a_max] + self.x.dot(self.x.T)
        self.ba[self.a_max] = self.ba[self.a_max] + r * self.x
        return

    def recommend(self, user_feature):
        blocks = self.blocks
        n_recom = self.n_recom
        Aa = self.Aa
        ba = self.ba
        user_features = np.tile(user_feature, (len(blocks), 1))
        #block_features = np.asmatrix(self.block_f.iloc[:, 1:].values)
        #arm_features = np.concatenate((user_features, block_features), axis=1)
        arm_features = user_features
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        i = 0
        for block in blocks:
            xa = arm_features[i].reshape(arm_features.shape[1], 1)
            AaI_temp = np.linalg.inv(Aa[block])
            theta_temp = AaI_temp.dot(ba[block])
            p_t[i] = theta_temp.T.dot(xa) + self.alpha * np.sqrt(xa.T.dot(AaI_temp).dot(xa))
            i += 1
        
        max_p_t = np.max(p_t)
        if max_p_t <= 0:
            print("max p_t={}, p_t={}".format(max_p_t, p_t))
        # I want to randomly break ties, np.argmax return the first occurence of maximum.
        # So I will get all occurences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to user t
        a_max = self.blocks[a_t]
        
        r_list = []
        for i in (np.argpartition(p_t.flatten(), -n_recom)[-n_recom:]):
            r_list.append(blocks[i])
        self.a_max = a_max
        self.x = arm_features[a_t].reshape(arm_features.shape[1], 1)
        return r_list

    def policy_evaluator(self):
        # read existing logfile
        blocks = self.blocks
        user_f = self.user_f
        rewards = []
        avg_rewards = []
        count_ = dict.fromkeys(blocks, 0)
        c = []
        new = dict.fromkeys(blocks, 0)
        save_dict('views.npy',{})
        views = load_dict('views.npy')
        g = os.walk("../../impression-2018")
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
                        recommends = self.recommend(user_feature)
                        #print(recommends)
                        count_[self.a_max] += 1
                        if href in recommends:
                            reward = 1
                            views[userid][self.a_max] = views[userid][self.a_max] + 1
                      
                            #rewards.append(1)
                            self.n += 1
                            self.mean_reward = self.mean_reward + (1 - self.mean_reward) / self.n
                            self.update(reward)
                        else:
                            x=np.array(list(views[userid].values()))
                            user_pos_rat= np.where(x!=0)[0]
                            num_known_ratings = len(user_pos_rat)+1
                          
                        # Find how much user likes the genre of the recommended movie based on his previous ratings.
                            likability = 0
                            a = pd.DataFrame.from_dict(views).T
                            for item in user_pos_rat:
                                likability += list(a.corrwith(a[self.a_max]).fillna(0))[item]
                            likability /= num_known_ratings
                            #print(likability)
                            binomial_reward_probability = likability
                            if binomial_reward_probability <= 0:
                            #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
                                binomial_reward_probability = 0 # this could be replaced by small probability
                            approx_rating = np.random.binomial(n=1, p=binomial_reward_probability)  # Bernoulli coin toss
                            if approx_rating == 1:
                                reward = 1
                                views[userid][self.a_max] = views[userid][self.a_max] + 1
                            else:
                                reward = 0
                            #rewards.append(0)
                            self.n += 1
                            self.mean_reward = self.mean_reward + (0 - self.mean_reward) / self.n
                            self.update(reward)
                        #avg_reward = np.average(np.array(rewards))
                        #views[userid][href] = 1
                        print(self.n,self.mean_reward)
                        #avg_rewards.append(avg_reward)
                        avg_rewards.append(self.mean_reward)
                        c.append(list(count_.values()))
        df = pd.DataFrame(c, columns=blocks)   
        self.reset()
        return avg_rewards,df

   
