## views through landing page
#find . -name '*.DS_Store' -type f -delete

import os
import random
import json
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def save_dict(filename,dictionary):
    np.save(filename,dictionary)

def load_dict(filename):
    dict = np.load(filename).item()
    return dict

block_f = pd.read_csv('../data/block_features/block_features.csv')
views = pd.read_csv('../data/views.csv')
block_f = block_f[block_f.blockid.isin(views.blockid.unique())]
blocks = list(block_f.blockid)

new_log_file = "../impression-2018"
g = os.walk(new_log_file) 
save_dict('views.npy',{})
new = dict.fromkeys(blocks, 0)
views = load_dict('views.npy')
record = {}
#load new file
for path,d,filelist in sorted(g):
    for filename in sorted(filelist):  
        file = open(os.path.join(path, filename))
        for line in file:
            l=json.loads(line) 
            href = l.setdefault('meta').setdefault('href')

            if href in list(blocks):
                #userid = l.setdefault('MediaGammaImpression', None)
                userid = l.setdefault('cookie').split(';')[0].split('=')[1]
                timestamp = l.setdefault('meta').setdefault('date').setdefault('date').split('.')[0].split('T')
                timestamp = int(time.mktime(time.strptime(timestamp[0] + ' ' + timestamp[1], "%Y-%m-%d %H:%M:%S")))
                if views == {}:
                    views[userid] = new.copy()
                else:
                    if userid not in views.keys():
                        views[userid] = new.copy()
                record[timestamp] = list(views[userid].values())
                views[userid][href] = views[userid][href] + 1
r = pd.DataFrame.from_dict(record, orient='index')
r = r.reset_index().rename(columns={'index':'userid'})
dummy = pd.read_csv("../data/user_features/user_features_dummy.csv")
x = pd.merge(r,dummy,on='userid')
r.to_csv("../data/user_features/user_features_log.csv",index=False,sep=',')
x.to_csv("../data/user_features/user_features.csv",index=False,sep=',')