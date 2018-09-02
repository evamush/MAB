import requests
import re
from lxml import html 
import pandas as pd
import time
import numpy as np

# "urls.csv" stores list of urls
start_url = list(pd.read_csv("../data/block_features/urls.csv",encoding = "utf-8")['URL'].values)

i = 1
#initialise a dict to store the crawled information
inform = {}
for url in start_url:
    inform[i] = {}
    response = requests.get(url)
    tree = html.fromstring(response.text)
    for j in range(1,7):
        cate = tree.xpath('//div[@class="travel-search-sidebar-list"][{}]/h5/text()'.format(j))
        cate = [re.sub("\n\t\t","",s) for s in cate]
        keys = tree.xpath('//div[@class="travel-search-sidebar-list"][{}]/ul/li/a/text()'.format(j))
        keys = [re.sub('\n\t\t', '', s) for s in keys]
        keys = [re.sub('\n\t', '', s) for s in keys]
        keys = [s for s in keys if s!='']
        values = tree.xpath('//div[@class="travel-search-sidebar-list"][{}]/ul/li/a/span/text()'.format(j))
        values = [re.sub('\D', '',s) for s in values]
        values = [int(s) for s in values]
        try:
            inform[i][cate[0]] = list(zip(keys, np.asarray(values)))
        except:
            pass
    #print(inform[i])
    i += 1

#save the crwaled data as 'block_f.npy'
np.save('../data/block_features/block_f.npy', inform) 

block_f = pd.DataFrame.from_dict(inform).T
block_f['blockid'] = start_url
#saved as csv file for future processing
block_f.to_csv("../data/block_features/block_f.csv",index=False,sep=',')

