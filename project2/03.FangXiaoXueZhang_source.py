#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a DIY module'

__author__ = 'Fang Linjiajie','Xue Zexiao','Xiao Ziliang','Zhang Wenyong'



import glob
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import seaborn as sns


class classifier(object):
    def __init__(self, data, target): # define function ,
        spsample = KFold(n_splits=2, random_state=234, shuffle=True)  # separate: half training, half test
        for trains_index, test_index in spsample.split(data):
            self.model, self.test = data[trains_index], data[test_index]    # use for modeling
            self.modelt, self.testt = target[trains_index], target[test_index]  # use for testing
        # just use self as a object for convenience

    def Kfold_validation(self):
        score = 0
        kf = KFold(n_splits=5, random_state=123, shuffle=True)
        for train_index, val_index in kf.split(self.model):
            # print("TRAIN:", train_index, "TEST:", test_index)
            data_train, data_valid = self.model[train_index], self.model[val_index]
            target_train, target_valid = self.modelt[train_index], self.modelt[val_index]
            # find training part and validation part here
            # Fit model with sample data
            self.cf.fit(data_train, target_train)
            score += 0.2*self.cf.score(data_valid, target_valid)
        return score

    def methods_selection(self, methods): #define function
        self.methods = methods
        score = np.array([])
        testscore = np.array([])
        trainscore = np.array([])
        alphahat = np.array([])

        if self.methods == 'randomforest':       # Pruning  Parameter Choosing
            for alpha in range(1, 20):
                self.cf = ensemble.RandomForestClassifier(n_estimators=alpha, random_state=120)  #puning parameter=number of trees
                testscore = np.insert(testscore, 0, self.test_score())
                trainscore = np.insert(trainscore, 0, self.train_score())
                score = np.insert(score, 0, self.Kfold_validation())
                alphahat = np.insert(alphahat, 0, alpha )
            alpha_index = score.argmax()

            trainscore = np.insert(trainscore[::-1], [0], [0])
            testscore = np.insert(testscore[::-1], [0], [0])
            score = np.insert(score[::-1],[0],[0])
            plt.plot(trainscore, label="Training Data")
            plt.plot(testscore, label="Test Data")
            plt.plot(score, label="5-Fold Cross-Validation")
            plt.xlabel('Number of Trees')
            plt.ylabel('Scores')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
            plt.show()
            self.cf = ensemble.RandomForestClassifier(n_estimators=int(alphahat[alpha_index])) # optimal model
            self.cf.fit(self.model, self.modelt)

        elif self.methods == 'bagging':
            self.cf = ensemble.RandomForestClassifier(max_features=None, random_state=120)

        elif self.methods == 'SVM':
            self.cf = svm.SVC()

        elif self.methods == 'logit':
            self.cf = LogisticRegression()

        else:
            print('Invalid method !')
            exit()



        if self.methods == 'randomforest':
            print('Random Forest--Optimal alpha:' + str(int(alphahat[alpha_index])))
            print('Random Forest Classifier:')
            print('Training Sore:' + str(self.train_score()))
            print('Score of 5-fold corss-validation\t' + str(score.max()))
            print('Test Score:' + str(self.test_score()))

        else:
            print(self.methods+'Classifier:')
            print('Training Sore:' + str(self.train_score()))
            print('Score of 5-fold corss-validation\t' + str(self.Kfold_validation()))
            print('Test Score:' + str(self.test_score()))

    def test_score(self): #out of sample score
        self.cf.fit(self.model, self.modelt)
        testscore = self.cf.score(self.test, self.testt)
        return testscore

    def train_score(self): #in sample score
        self.cf.fit(self.model, self.modelt)
        trainscore = self.cf.score(self.model, self.modelt)
        return trainscore


    def scatter_inputdata(self):
        colors_filter = self.modelt[np.array(range(len(data_trains)//3-1))*3]
        data_filter = self.model[np.array(range(len(data_trains)//3-1))*3]
        x = TSNE(random_state=123).fit_transform(data_filter)
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 10))
        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                        c=palette[colors_filter.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        # We add the labels for each digit.
        txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(x[colors_filter == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
        return f, ax, sc, txts


if __name__ == "__main__":

    path = r'/Users/fanglinjiajie/Documents/Python/handwriting'  # use your own path
    allFiles = glob.glob(path + "/*.csv")
    train = []
    for file_ in allFiles:
        df = pd.read_csv(file_, header=None)
        train.append(df)

    data = train[0].values
    target = np.zeros(train[0].shape[0])
    for i in range(1, len(train)):
        data = np.concatenate((data, train[i].values), axis=0)
        temp = np.arange(train[i].shape[0])
        temp[:] = i
        target = np.concatenate((target, temp), axis=0)

    spsample = KFold(n_splits=2, random_state=234, shuffle=True) # separate: half training, half test
    for trains_index, test_index in spsample.split(data):
        data_trains, data_test = data[trains_index], data[test_index]
        target_trains, target_test = target[trains_index], target[test_index]

    cf = classifier(data, target)

    print("Please wait a second for input data visualization...")
    cf.scatter_inputdata()
    h = .02

    #====================== plot classification
    names = ['randomforest', 'bagging', 'SVM', 'logit']

    classifiers = [ensemble.RandomForestClassifier(max_leaf_nodes=14, random_state=120),
                   ensemble.RandomForestClassifier(max_leaf_nodes=None, random_state=120),
                   svm.SVC(),
                   LogisticRegression(), ]

    colors_filter = target[np.array(range(len(data) // 3 - 1)) * 3]
    data_filter = data[np.array(range(len(data) // 3 - 1)) * 3]
    X = TSNE(random_state=123).fit_transform(data_filter)
    y = colors_filter
    linearly_separable = (X, y)
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_valid = X[train_index], X[test_index]
        target_train, target_valid = y[train_index], y[test_index]
    x_min, x_max = data_train[:, 0].min() - .5, data_train[:, 0].max() + .5
    y_min, y_max = data_train[:, 1].min() - .5, data_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for i in range(4):
        classifiers[i].fit(data_train, target_train)

        palette = np.array(sns.color_palette("hls", 10))
        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, \
                   c=palette[colors_filter.astype(np.int)])

        Z = classifiers[i].predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = classifiers[1].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.3)
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        #====================
    plt.show()

    methods = 'randomforest'

    while methods == 'randomforest'or'SVM'or'bagging'or'logit':
        methods = input('Enter the classifier methods:\'randomforest\',\'bagging\',\'SVM\' ,\'logit\'or arbitrary key for quit\n')
        if methods != 'randomforest'or'SVM'or'bagging'or'logit':
            print('Invalid methods!')
            exit()
        cf = classifier(data, target)
        meth = cf.methods_selection(methods)







