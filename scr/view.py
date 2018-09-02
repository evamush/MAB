## views through landing page
#find . -name '*.DS_Store' -type f -delete

import os
import random
import json
import time
import pandas as pd
import numpy as np

block = pd.read_csv("../data/block_features/blocks.csv")
block = block[block['Collection '].isnull()==False]
block = block[block['URL'].isnull()==False]
block = block.set_index(np.arange(len(block)))
originalblock = list(block.URL)

new_log_file = "../impression-2018"
g = os.walk(new_log_file) 

newview = []

#load new file
   
for path,d,filelist in sorted(g):
    for filename in sorted(filelist):  
        file = open(os.path.join(path, filename))
        for line in file:
            l=json.loads(line) 
            href = l.setdefault('meta').setdefault('href')

            if href in list(originalblock):
                #userid = l.setdefault('MediaGammaImpression', None)
                #userid = l.setdefault('cookie').split(';')[0].split('=')[1]
                timestamp = l.setdefault('meta').setdefault('date').setdefault('date').split('.')[0].split('T')
                timestamp = int(time.mktime(time.strptime(timestamp[0] + ' ' + timestamp[1], "%Y-%m-%d %H:%M:%S")))
                newview.append([timestamp,href])

print(len(newview))    
newview=pd.DataFrame(newview,columns=['userid','blockid'])
newview.to_csv("../data/views.csv",index=False,sep=',')
             
