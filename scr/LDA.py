#LDA
import gensim
from gensim import corpora
import string
import re
import io
import math
import pandas as pd
import numpy as np

block = pd.read_csv("../data/block_features/tokenized.csv")
block_clean = [block.split() for block in block.tokenized] 
dictionary = corpora.Dictionary(block_clean)
doc_term_matrix = [dictionary.doc2bow(block) for block in block_clean]
# creation and the running of the model on the created matrix
Lda = gensim.models.ldamodel.LdaModel

def perplexity(ldamodel, doc_term_matrix, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    #print ('the info of this ldamodel: \n')
    #print ('size_dictionary: %s; num of topics: %s'%(size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for i in doc_term_matrix:
        doc_topics_ist.append(ldamodel.get_document_topics(i, minimum_probability=0))    
    testset_word_num = 0
    for i in range(len(doc_term_matrix)):
        prob_doc = 0.0 # the probablity of the doc
        doc = doc_term_matrix[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in doc:
            prob_word = 0.0 # the probablity of the word 
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    #print ("the perplexity of this ldamodel is : %s"%prep)
    return prep
'''
p=[]
for i in range(3,100):
    ldamodel=Lda(doc_term_matrix, num_topics=i, id2word = dictionary, passes=5,minimum_probability=0)
    p.append(perplexity(ldamodel,doc_term_matrix, dictionary, len(dictionary), i))


plt.plot(p)
plt.xlabel('no. topics')
plt.ylabel('perplexity')
plt.show()
from itertools import chain

lda_corpus = ldamodel[doc_term_matrix]
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
#print(threshold)
clusters = []
for topic in range(n_topics):
    #print(lda_corpus[topic])
    clusters.append([list(block.tokenized).index(j) for i,j in zip(lda_corpus,list(block.tokenized)) if i[topic][1] > threshold])
clusters
'''

n_topics = 70
ldamodel=Lda(doc_term_matrix, num_topics=n_topics, id2word = dictionary, passes=5,minimum_probability=0)
doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
for i in doc_term_matrix:
    doc_topics_ist.append(ldamodel.get_document_topics(i, minimum_probability=0))    

items_features=[]
for i in range(len(block)):
    items_features.append([block.blockid[i]] + [v for k, v in doc_topics_ist[i]])
block_features = pd.DataFrame(items_features,columns=['blockid'] + list(range(70)))
