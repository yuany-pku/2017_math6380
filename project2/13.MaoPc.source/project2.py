#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.decomposition import NMF
import itertools
from sklearn import manifold

import random 
random.seed(10)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

n_features = 1000
n_topics = 8
n_top_words = 10

df = pd.read_csv("nips/papers.csv").sort(['year'])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print ""

tfidf_vectorizer = TfidfVectorizer(max_df=0.95,min_df=2,max_features=n_features,stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['paper_text'])
nmf = NMF(n_components=n_topics, random_state=0,alpha=.1, l1_ratio=.5).fit(tfidf)
print("Topics found via NMF:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

nmf_embedding = nmf.transform(tfidf)
nmf_embedding = (nmf_embedding - nmf_embedding.mean(axis=0))/nmf_embedding.std(axis=0)
top_idx = np.argsort(nmf_embedding,axis=0)[-3:]

count = 0
for idxs in top_idx.T: 
    print("\nTopic {}:".format(count))
    for idx in idxs:
        print(df.iloc[idx]['title'])
    count += 1
    
df['t'] = nmf_embedding.argmax(axis=1)


time_step = ['1987-1991', '1992-1996' ,'1997-2001', '2002-2006', '2007-2011', '2012-2016']
for year, idx in zip([1991,1996,2001,2006,2011,2016], range(6)):
    data = df[df['year']<=year]
    num = []
    for x in range(8):
        num.append(sum(data['t']==x))
    #sns.barplot(x_label, np.array(num), palette="Blues_d", ax=ax1[idx])
    #ax1[idx].set_ylabel("# of papers")
    fig, ax = plt.subplots()
    plt.bar(range(8), num)
    plt.xticks(range(8), ('t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'))
    ax.set_ylabel('# of papers')
    ax.set_xlabel('Topics')
    ax.set_title('Published between '+time_step[idx])
    plt.show()
    


'''topic 0: optimization algorithms
topic 1: neural network application
topic 2: reinforcement learning
topic 3: bayesian methods
topic 4: image recognition
topic 5: artificial neuron design
topic 6: graph theory
topic 7: kernel methods'''
line = ['b','g','r','c','m','y','k','#8487bc']
all_tp = []
for x in range(8):
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
ax.set_ylabel('# of papers')
ax.set_xlabel('Year')
ax.set_title('Accepted papers between 1987-2016')
plt.legend(loc='best')
plt.show()
##############################################
train_data = df[(df['year']!=2016)&(df['year']!=2015)]
test_data = df[(df['year']==2016)|(df['year']==2015)]
label_name = ['t1','t2','t3','t4','t5','t6','t7','t8']
#### naive_bayes
text_clf_1 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
_ = text_clf_1.fit(train_data['paper_text'], train_data['t'])
predicted_1 = text_clf_1.predict(test_data['paper_text'])
print "naive_bayes: ", np.mean(predicted_1 == test_data['t'])

print(metrics.classification_report(test_data['t'], predicted_1,
    target_names=label_name))
c_matrix_1 = metrics.confusion_matrix(test_data['t'], predicted_1)
print c_matrix_1
########################
'''naive_bayes:  0.555670103093
             precision    recall  f1-score   support

         t1       0.80      0.83      0.82       351
         t2       0.71      0.40      0.51        75
         t3       1.00      0.03      0.05        73
         t4       0.30      1.00      0.46       147
         t5       1.00      0.19      0.32        99
         t6       1.00      0.43      0.61        23
         t7       1.00      0.01      0.02       103
         t8       0.95      0.37      0.54        99

avg / total       0.79      0.56      0.51       970

[[293   0   0 ...,   0   0   0]
 [  2  30   0 ...,   0   0   0]
 [ 21   3   2 ...,   0   0   0]
 ..., 
 [  1   0   0 ...,  10   0   0]
 [ 27   0   0 ...,   0   1   1]
 [ 22   0   0 ...,   0   0  37]]'''
##################################
### SVM
text_clf_2 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)), ])

_ = text_clf_2.fit(train_data['paper_text'], train_data['t'])
predicted_2 = text_clf_2.predict(test_data['paper_text'])
print "svm: ", np.mean(predicted_2 == test_data['t'])

print(metrics.classification_report(test_data['t'], predicted_2,
    target_names=label_name))
c_matrix_2 = metrics.confusion_matrix(test_data['t'], predicted_2)
print c_matrix_2
#####################################
'''svm:  0.920618556701
             precision    recall  f1-score   support

         t1       0.91      0.98      0.94       351
         t2       0.91      0.92      0.91        75
         t3       0.92      0.90      0.91        73
         t4       0.92      0.90      0.91       147
         t5       0.93      0.96      0.95        99
         t6       0.84      0.91      0.87        23
         t7       0.97      0.86      0.91       103
         t8       0.95      0.78      0.86        99

avg / total       0.92      0.92      0.92       970

[[344   0   0 ...,   0   1   1]
 [  1  69   0 ...,   0   1   1]
 [  4   1  66 ...,   2   0   0]
 ..., 
 [  1   0   0 ...,  21   0   0]
 [  7   0   3 ...,   0  89   0]
 [ 17   1   0 ...,   0   0  77]]'''
####################################
### Random Forest
text_clf_3 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=35)), ])

_ = text_clf_3.fit(train_data['paper_text'], train_data['t'])
predicted_3 = text_clf_3.predict(test_data['paper_text'])
print "Random Forest: ", np.mean(predicted_3 == test_data['t'])

print(metrics.classification_report(test_data['t'], predicted_3,
    target_names=label_name))
c_matrix_3 = metrics.confusion_matrix(test_data['t'], predicted_3)
print c_matrix_3
#####################################
'''Random Forest:  0.715463917526
             precision    recall  f1-score   support

         t1       0.76      0.89      0.82       351
         t2       0.71      0.53      0.61        75
         t3       0.84      0.71      0.77        73
         t4       0.61      0.82      0.70       147
         t5       0.77      0.73      0.75        99
         t6       0.67      0.70      0.68        23
         t7       0.74      0.36      0.48       103
         t8       0.60      0.45      0.52        99

avg / total       0.72      0.72      0.70       970

[[312   1   4 ...,   0   5  10]
 [  8  40   0 ...,   2   0   6]
 [ 12   2  52 ...,   0   0   0]
 ..., 
 [  1   0   0 ...,  16   0   0]
 [ 37   1   3 ...,   1  37   6]
 [ 29   1   1 ...,   0   4  45]]'''
#######################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(b=False)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['t1','t2','t3','t4','t5','t6','t7','t8']
plt.figure()
plot_confusion_matrix(c_matrix_1, classes=class_names,
                          title='Confusion matrix (Naive Bayes)')
plt.figure()
plot_confusion_matrix(c_matrix_2, classes=class_names,
                          title='Confusion matrix (SVM)')
plt.figure()
plot_confusion_matrix(c_matrix_3, classes=class_names,
                          title='Confusion matrix (Random Forest)')
