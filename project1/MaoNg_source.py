#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:24:42 2017

@author: root
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

n_features = 1000
n_topics = 10
n_top_words = 8

df = pd.read_csv("nips/papers.csv").sort(['year'])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print ""


tfidf_vectorizer = TfidfVectorizer(max_df=0.95,min_df=2,max_features=n_features,stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['paper_text'])
svd = TruncatedSVD(n_components=n_topics).fit(tfidf)
print("Topics found via SVD:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(svd, tfidf_feature_names, n_top_words)

svd_embedding = svd.transform(tfidf)
svd_embedding = (svd_embedding - svd_embedding.mean(axis=0))/svd_embedding.std(axis=0)
top_idx = np.argsort(svd_embedding,axis=0)

count = 0
for idxs in top_idx[-3:].T: 
    print("\nTopic {}:".format(count))
    for idx in idxs:
        print(df.iloc[idx]['title'])
    count += 1
    
#tsne = TSNE(random_state=3211)
#tsne_embedding = tsne.fit_transform(svd_embedding)
#tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])
df['t'] = svd_embedding.argmax(axis=1)



#f, ax1 = plt.subplots(6, 1, figsize=(8, 6), sharex=True)
#x_label = np.array(['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10'])
time_step = ['1987-1991', '1992-1996' ,'1997-2001', '2002-2006', '2007-2011', '2012-2016']
for year, idx in zip([1991,1996,2001,2006,2011,2016], range(6)):
    data = df[df['year']<=year]
    num = []
    for x in range(10):
        num.append(sum(data['t']==x))
    #sns.barplot(x_label, np.array(num), palette="Blues_d", ax=ax1[idx])
    #ax1[idx].set_ylabel("# of papers")
    fig, ax = plt.subplots()
    plt.bar(range(10), num)
    plt.xticks(range(10), ('t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10'))
    ax.set_ylabel('# of papers')
    ax.set_xlabel('Topics')
    ax.set_title('Published between '+time_step[idx])
    plt.show()
#sns.despine(bottom=False)
#plt.setp(f.axes, yticks=[])
#plt.tight_layout(h_pad=6)
line = ['b','g','r','c','m','y','k','k:','b:','m-.']
all_tp = []
for x in range(10):
    data = df[df['t'] == x]
    num = []
    for year in range(1987,2017):
        num.append(sum(data['year']==year))
    all_tp.append(num)
year_range = range(1987,2017)
fig, ax = plt.subplots()
plt.plot(year_range, all_tp[0], line[0],label='Topic 1',linewidth=1.5)
plt.plot(year_range, all_tp[1], line[1],label='Topic 2',linewidth=1.5)
plt.plot(year_range, all_tp[2], line[2],label='Topic 3',linewidth=1.5)
plt.plot(year_range, all_tp[3], line[3],label='Topic 4',linewidth=1.5)
plt.plot(year_range, all_tp[4], line[4],label='Topic 5',linewidth=1.5)
plt.plot(year_range, all_tp[5], line[5],label='Topic 6',linewidth=1.5)
plt.plot(year_range, all_tp[6], line[6],label='Topic 7',linewidth=1.5)
plt.plot(year_range, all_tp[7], line[7],label='Topic 8',linewidth=1.5)
plt.plot(year_range, all_tp[8], line[8],label='Topic 9',linewidth=1.5)
plt.plot(year_range, all_tp[9], line[9],label='Topic 10',linewidth=1.5)
ax.set_ylabel('# of papers')
ax.set_xlabel('Year')
ax.set_title('Accepted papers between 1987-2016')
plt.legend(loc='best')
plt.show()

# TODO Lasso
#for x in range(10):
#    data = df[df['t'] == x]
#    data = data[df['year']<=2013]
#    num = []
#    for year in range(1987,2017):
#        num.append(sum(data['year']==year))
#    all_tp.append(num)


        
    
