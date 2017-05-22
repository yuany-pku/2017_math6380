# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:44:12 2017

@author: isaac_xwq
"""
import random
import pandas as pd
import time
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm,metrics
from matplotlib import pyplot as plt
#from sklearn import model_selection
#from sklearn import metrics
from sklearn import cross_validation,manifold
from sklearn.grid_search import GridSearchCV
###load data from train0-9
filename=list()
for file in os.listdir("D:/Study/xwq/XWQ/ustmafs/math6380/finalproject/digit"):
    if file.endswith(".txt"):
        filename.append(file)
train=pd.DataFrame()
for i in range(len(filename)):   ###the last column is type of digit
    temp=pd.read_table("D:/Study/xwq/XWQ/ustmafs/math6380/finalproject/digit/"+filename[i],sep=',',header=None)
    temp['digit']=i
    train=train.append(temp)
test=pd.read_csv("D:/Study/xwq/XWQ/ustmafs/math6380/finalproject/zip.test",header=None,sep=' ') ##the first col is type of digit
####    
def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['IC50s'])

    #Predict training set:
    #dtrain_predictions = alg.predict(dtrain[predictors])
   # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['IC50s'], cv=cv_folds,scoring='neg_mean_squared_error')

    #Print model report:
    print ("\nModel Report")
   # print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['IC50s'].values, dtrain_predictions))
  #  print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['IC50s'], dtrain_predprob))

    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
#####SVM
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(train.iloc[::,0:256], train['digit'])

SVM_classifier = svm.SVC(C=10,kernel='rbf')
SVM_classifier.fit(train.iloc[::,0:256], train['digit'])

# Now predict the value of the digit on the second half:
expected = test[0]
predicted = SVM_classifier.predict(test.iloc[::,1:257])
print("Classification report for classifier %s:\n%s\n"
      % (SVM_classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#########
######randomforest

###forest puning
start=time.clock()
para_test1={'n_estimators':np.arange(100,201,10),'max_depth':np.arange(2,10,1),'min_samples_split':np.arange(3,10,1),'min_samples_leaf':np.arange(1,10,1),}
gsearch1=GridSearchCV(estimator=RandomForestClassifier(max_features='sqrt'),param_grid=para_test1,cv=5)
gsearch1.fit(train.iloc[::,0:256],train['digit'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
time=time.clock()-start
####max_depth:9 min_samples_leaf:2  min_sample_split:3 n_estimators:180
###determinate max_depth and min_sample_split, min_sample_leaf, max_feature
###best {'max_depth': 4, 'min_samples_split': 7}, 
RF_classifier=RandomForestClassifier(n_estimators=180,max_depth=9,min_samples_leaf=2,min_samples_split=3)
RF_classifier.fit(train.iloc[::,0:256], train['digit'])   
expected = test[0]
predicted =RF_classifier.predict(test.iloc[::,1:257])
print("Classification report for classifier %s:\n%s\n"
      % (RF_classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#


for index in range(4):
    plt.subplot(2, 4, index+5)
    plt.axis('off')
    i=random.randint(0,2007)
    image=np.reshape(test.iloc[i,0:256],(16,16))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % predicted[i])

plt.show()

####load data from train0-9

###
