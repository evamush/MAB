import numpy as np
import pandas as pd
import os
import json
import time
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random

def save_dict(filename,dictionary):
    np.save(filename,dictionary)

def load_dict(filename):
    dict = np.load(filename).item()
    return dict

def update_dict(new,old):
	"""
    used to update the user views history
    new with all items set to zero

    """
    new.update(old)
    return new

class HybridLinUCB:
	"""
    HybridLinUCB
    alpha: control the exploration and exploitation
    user_f: the file storing the user features
    block_f: the file storing the user features
    n_recom: number of recommendations
    """
    def __init__(self,alpha,user_f,block_f, n_recom):
        self.alpha = alpha  # explore depth
        self.block_f = block_f
        self.n_recom = n_recom
        self.user_f = user_f
        self.mean_reward = 0
        self.n = 0
        self.z = None
        #self.d = self.user_f.shape[1] + self.block_f.shape[1] - 2 
        # dimension of user features when 'log feature' is included = 'Disjoint LinUCB'
        self.d = self.user_f.shape[1] - 1
        # dimension of user features when 'log feature' is excluded = 'Tied LinUCB'
        self.k = self.d * self.d  # dimension of article features = k
        self.Aa = {}
        self.ba = {}
        self.B = {}
        # collection of vectors to compute disjoint part, d*1
        self.a_max = 0
        self.x = None
        #self.theta = {}   linear parameter
        self.blocks = list(block_f.blockid)
        self.A0 = np.identity(self.k, dtype=float)
        self.b0 = np.zeros((self.k,1), dtype=float)

        for key in self.blocks:
            self.Aa[key] = np.identity(self.d, dtype=float)
            self.ba[key] = np.zeros((self.d, 1), dtype=float)
            self.B[key] = np.zeros((self.d, self.k), dtype=float)

    def reset(self):
        self.A0 = np.identity(self.k, dtype=float)
        self.b0 = np.zeros((self.k,1), dtype=float)

        for key in self.blocks:
            self.Aa[key] = np.identity(self.d, dtype=float)
            self.ba[key] = np.zeros((self.d, 1), dtype=float)
            self.B[key] = np.zeros((self.d, self.k), dtype=float)
        return
        
    def update(self, r):
        a_max = self.a_max
        z = self.z
        A_at_inv = np.linalg.inv(self.Aa[a_max])
        self.A0 = self.A0 + self.B[a_max].T.dot(A_at_inv).dot(self.B[a_max])
        self.b0 = self.b0 + self.B[a_max].T.dot(A_at_inv).dot(self.ba[a_max].reshape(-1, 1))
        self.B[a_max] = self.B[a_max] + self.x.dot(z.T)
        self.Aa[a_max] = self.Aa[a_max] + self.x.dot(self.x.T)
        self.ba[a_max] = self.ba[a_max] + r * self.x
        self.A0 = self.A0 + z.dot(z.T) - self.B[a_max].T.dot(A_at_inv).dot(self.B[a_max])
        self.b0 = self.b0 + r * z - self.B[a_max].T.dot(A_at_inv).dot(self.ba[a_max].reshape(-1, 1))

    def recommend(self, user_feature):
        blocks = self.blocks
        n_recom = self.n_recom
        Aa = self.Aa
        B = self.B
        ba = self.ba
        #print(self.A0)
        A0I = np.linalg.pinv(self.A0)
        #print(np.allclose())
        beta = A0I.dot(self.b0)

        user_features = np.tile(user_feature, (len(blocks), 1))
        block_features = np.asmatrix(self.block_f.iloc[:, 1:].values)
        '''
        arm_features = np.concatenate((user_features, block_features), axis=1)
        pca = PCA(n_components=self.d)
        arm_features = pca.fit_transform(arm_features)
        '''
        arm_features = user_features
        p_t = np.zeros(shape=(len(blocks),), dtype=float)

        i = 0
        for block in blocks:
            xa = arm_features[i].reshape(-1, 1)
            #xa = np.array([user_feature]).reshape(-1, 1)
            za = np.outer(block_features[i].reshape(-1,1), arm_features[i]).flatten().reshape(self.k,1)
            #za = np.outer(block_features[i].reshape(-1,1), xa).flatten().reshape(self.k,1)
            
            AaI_temp = np.linalg.inv(Aa[block])
            ba_temp = ba[block].reshape(-1, 1)
            theta_temp = AaI_temp.dot(ba_temp - B[block].dot(beta))

            sa = za.T.dot(A0I).dot(za) - 2 * za.T.dot(A0I).dot(B[block].T).dot(AaI_temp).dot(xa)
            sa += xa.T.dot(AaI_temp).dot(xa) + xa.T.dot(AaI_temp).dot(B[block]).dot(A0I).dot(B[block].T).dot(AaI_temp).dot(xa)

            p_t[i] = (za.T.dot(beta) + xa.T.dot(theta_temp))+ self.alpha * np.sqrt(sa)
            i += 1
        max_p_t = np.max(p_t)

        # randomly break ties, np.argmax return the first occurence of maximum.
        # get all occurences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to user t
        # self.x = arm_features[a_t].reshape(arm_features[a_t].shape[1], 1)
        a_max = blocks[a_t]
        x = arm_features[a_t].reshape(-1, 1)
        #x = np.array([user_feature]).reshape(-1, 1)
        self.z = np.outer(block_features[a_t].reshape(-1,1), arm_features[a_t]).flatten().reshape(self.k,1)
        #self.z = np.outer(block_features[a_t].reshape(-1,1), x).flatten().reshape(self.k,1)
        
        r_list = []
        for i in (np.argpartition(p_t.flatten(), -n_recom)[-n_recom:]):
            r_list.append(blocks[i])

        self.a_max = a_max
        self.x = x
        return r_list
 
    def policy_evaluator(self, views_f):
        # read existing logfile
        blocks = self.blocks
        user_f = self.user_f
        rewards = []
        avg_rewards = []
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
            #if sum(user_feature)!=0:
            recommends = self.recommend(user_feature)
            if href in recommends:
                reward = 1
            else:
                reward = 0

            self.n += 1
            self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
            self.update(reward)
            avg_rewards.append(self.mean_reward)
            print(self.mean_reward)
        self.reset()
        return avg_rewards
