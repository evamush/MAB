import numpy as np
import pandas as pd
import os
import json
import time
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing

opt = 'https://www.petspyjamas.com/travel/nav/dog-friendly-cottages-hot-tub'

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
    """
    LinUCB: for 'Cheat LinUCB'
    alpha: control the exploration and exploitation
    user_f: the file storing the user features
    block_f: the file storing the user features
    n_recom: number of recommendations
    """
    def __init__(self,alpha,user_f, block_f,n_recom):
        self.alpha = alpha  # explore depth
        self.block_f = block_f
        self.user_f = user_f
        self.n_recom = n_recom
        self.mean_reward = 0
        #self.d = self.user_f.shape[1] + self.block_f.shape[1] - 2 
        # dimension of user features when 'log feature' is included = 'Disjoint LinUCB'
        self.d = self.user_f.shape[1] - 1
        # dimension of user features when 'log feature' is excluded = 'Tied LinUCB'
        
        self.Aa = {}
        # collection of matrix to compute disjoint part of each block a d*d
        self.ba = {}
        # collection of vectors to compute disjoint part, d*1

        self.a_max = 0
        self.x = None
        self.n = 0 #store the temporal chosen arm
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
        #1. based on Disjoint LinUCB
        arm_features = user_features

        #2. based on Tied LinUCB
        #block_features = np.asmatrix(self.block_f.iloc[:, 1:].values)
        #arm_features = np.concatenate((user_features, block_features), axis=1)
        
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        #initialise the array of expected reward
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
        # randomly break ties, np.argmax return the first occurence of maximum.
        # get all occurences of the max and randomly select between them
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
                        if href in recommends:
                            reward = 1
                            views[userid][self.a_max] = views[userid][self.a_max] + 1
                            self.n += 1
                            self.mean_reward = self.mean_reward + (1 - self.mean_reward) / self.n
                            self.update(reward)
                        else:
                            x=np.array(list(views[userid].values()))
                            user_pos_rat= np.where(x!=0)[0]
                            num_known_ratings = len(user_pos_rat)+1
                          
                        # Find how much user likes the recommended arm based on his previous ratings.
                            likability = 0
                            a = pd.DataFrame.from_dict(views)
                            # set a threshold considering the processing time
                            if len(a)>2000:
                                a = a.iloc[:,:2000].T
                            else:
                                a=a.T
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
                                print('cheat') # give cheat reward if the recommended item is underestimated
                                reward = 1
                                views[userid][self.a_max] = views[userid][self.a_max] + 1
                            else:
                                reward = 0
                            self.n += 1
                            self.mean_reward = self.mean_reward + (0 - self.mean_reward) / self.n
                            self.update(reward) #update the strategy according to the cheat reward
                        #print(self.n,self.mean_reward)
                        avg_rewards.append(self.mean_reward) # append the real mean reward
        self.reset()
        return avg_rewards

