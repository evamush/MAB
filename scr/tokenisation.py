import pandas as pd
import numpy as np
import requests
import bs4
import re, nltk
from sklearn import feature_extraction
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lem = WordNetLemmatizer()

#remove numbers from tokens
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def tokenization(text):
    text = " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()
    lemmas = [lem.lemmatize(t) for t in nltk.word_tokenize(text)]
    #nonstopwords = [w for w in lemmas if w not in feature_extraction.text.ENGLISH_STOP_WORDS]
    stop = stopwords.words('english')
    nonstopwords = [w for w in lemmas if w not in stop]
    nonstopwords = [w for w in nonstopwords if not w.isdigit()]
    nonstopwords = [w for w in nonstopwords if not hasNumbers(w)]
    text = " ".join(nonstopwords) 
    return text

block = pd.read_csv("../data/block_features/urls.csv",encoding = "utf-8")
blocks = list(pd.read_csv("../data/block_features/urls.csv",encoding = "utf-8")['URL'].values)

a=[]
for i in range(len(blocks)):
    url = blocks[i]
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text,"lxml")
    #title = soup.title.string
    words = ''
    if blocks[i]=='Spotlight on Cotswolds':
        for x in soup.find_all('p'):
            words += (x.string)
            words += ' '
        for y in soup.find_all('h3'):
            if y.string!= None:
                words += (y.string)
                words += ' '
        for z in soup.find_all('h4'):
            words += (z.string)
            words += ' '
    else:
        for x in soup.find_all(attrs={"itemprop":"description"}):
            words += (x['content'])
            words += ' '

        for y in soup.find_all(attrs={"itemprop":"name"})[1:]:
            words += (y['content'])
            words += ' '
    a.append(words)

block['tokenized']=a  

for i in range(len(block)):
    block.loc[i,("tokenized")] = tokenization(block.loc[i,("tokenized")])

block = block[['URL','tokenized']]
block = block.rename(columns={'URL':'blockid'})
block.to_csv("../data/block_features/tokenized.csv",index=False,sep=',')
