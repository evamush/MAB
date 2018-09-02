import os
import random
import json
import time
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

block = pd.read_csv("../data/block_features/blocks.csv")
block = block[block['Collection '].isnull()==False]
block = block[block['URL'].isnull()==False]
block = block.set_index(np.arange(len(block)))
blocks = list(block.URL)

new_log_file = "../impression-2018"
g = os.walk(new_log_file) 

features = []

for path,d,filelist in sorted(g):
    for filename in sorted(filelist):  
        file = open(os.path.join(path, filename))
        for line in file:
            l=json.loads(line) 
            href = l.setdefault('meta').setdefault('href')
            if href in list(blocks):
                timestamp = l.setdefault('meta').setdefault('date').setdefault('date').split('.')[0].split('T')
                timestamp = int(time.mktime(time.strptime(timestamp[0] + ' ' + timestamp[1], "%Y-%m-%d %H:%M:%S")))
                timezone = l.setdefault('meta').setdefault('date').setdefault('timezoneOffset',None)
                if (timezone == -180 or timezone == -120 or timezone == -60 or timezone == 0 or timezone == 240 or timezone == -420):
                 
                    timezone = str(timezone)
                else:
                    timezone = 'timezone_other'
              
                language = l.setdefault('meta').setdefault('navigator',None).setdefault('language',None)
         
                if (language == 'en-GB' or language == 'en-gb'):
                    language = 'en-GB'
                elif (language =='en-US' or language =='en-us'):
                    language = 'en-US'
                else:
                    language = 'language_other'
        
                height = l.setdefault('meta').setdefault('screen').setdefault('height',None)
                width = l.setdefault('meta').setdefault('screen').setdefault('width',None)
                if (height!=0 and width!=0):
                    screen= round(height/width,2)
                else:
                    screen = 0
                if screen < 0.55:
                    screen = 'smallratio'
                elif screen >=1.86:
                    screen = 'largeratio'
                else:
                    screen = screen
        
                platform = l.setdefault('meta').setdefault('navigator',None).setdefault('platform',None)
                
                features.append([timestamp,timezone,language,platform,screen])
                
userfeatures = pd.DataFrame(features,columns=['userid','timezone','language','platform','screensize'])
userfeatures.to_csv("../data/user_features/user_features_raw.csv",index=False,sep=',')
userfeatures= pd.get_dummies(userfeatures,columns=['platform','language','timezone','screensize'])
userfeatures.to_csv("../data/user_features/user_features_dummy.csv",index=False,sep=',')



