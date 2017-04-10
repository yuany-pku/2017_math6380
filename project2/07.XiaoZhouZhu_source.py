# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:54:07 2017

@author: Isaac_xwq
"""
import pandas as pd
import numpy as np
#import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
#from sklearn import model_selection
#from sklearn import metrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

dt=pd.read_csv("F:/xwq/XWQ/ustmafs/math6380/miniproject2/data/drug1/OneDrug_train.csv")
dtest=dt.iloc[np.where(dt.isnull().T.any())]
dt=dt.dropna(axis=0,how='any')
#dt1=dt.iloc[::,1:21]
#y=dt.iloc[::,21]

rf=RandomForestRegressor()
       
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

####
predictors=dt.columns[3:63]
gbm1=GradientBoostingRegressor()
modelfit(gbm1,dt,predictors)
rf1=RandomForestRegressor()
modelfit(rf1,dt,predictors)
####determinate n_estimators   
####best {'n_estimators': 30}, 
para_test1={'n_estimators':np.arange(40,201,10)}
gsearch1=GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,min_samples_split=5,min_samples_leaf=1,max_depth=3,max_features='sqrt'),param_grid=para_test1,cv=5,scoring='neg_mean_squared_error')
gsearch1.fit(dt[predictors],dt['IC50s'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
###determinate max_depth and min_sample_split, min_sample_leaf, max_feature
###best {'max_depth': 4, 'min_samples_split': 7}, 
para_test2={'max_depth':np.arange(2,10,1),'min_samples_split':np.arange(3,10,1)}
gsearch2=GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,n_estimators=30,max_features='sqrt'),param_grid=para_test2,cv=5,scoring='neg_mean_squared_error')
gsearch2.fit(dt[predictors],dt['IC50s'])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

para_test3={'min_samples_leaf':np.arange(1,10,1)}
gsearch3=GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,n_estimators=30,max_depth=4,min_samples_split=7,max_features='sqrt'),param_grid=para_test3,cv=5,scoring='neg_mean_squared_error')
gsearch3.fit(dt[predictors],dt['IC50s'])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#####best{'min_samples_leaf': 5},
para_test4={'max_features':np.arange(5,30,2)}
gsearch4=GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,n_estimators=30,max_depth=4,min_samples_split=7,min_samples_leaf=5),param_grid=para_test4,cv=5,scoring='neg_mean_squared_error')
gsearch4.fit(dt[predictors],dt['IC50s'])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#### best {'max_features': 19},
####adjust subsample 
para_test5={'subsample':np.arange(0.6,1,0.05)}
gsearch5=GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,n_estimators=30,max_depth=4,min_samples_split=7,min_samples_leaf=5,max_features=19),param_grid=para_test5,cv=5,scoring='neg_mean_squared_error')
gsearch5.fit(dt[predictors],dt['IC50s'])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
####best {'subsample': 0.8500000000000002},
####determinate learning_rate
para_test6={'learning_rate':np.arange(0.005,0.105,0.005)}
gsearch6=GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=30,max_depth=4,min_samples_split=7,min_samples_leaf=5,max_features=19,subsample=0.85),param_grid=para_test6,cv=5,scoring='neg_mean_squared_error')
gsearch6.fit(dt[predictors],dt['IC50s'])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
###{'learning_rate': 0.090000000000000011}
gbm2=GradientBoostingRegressor(learning_rate=0.09,n_estimators=30,max_depth=4,min_samples_split=7,min_samples_leaf=5,max_features=19,subsample=0.85)
modelfit(gbm2,dt,predictors)
gbm22=gbm2.fit(dt[predictors],dt['IC50s'])
ytest=gbm22.predict(dtest[predictors])
dtest.iloc[::,2]=ytest
result=dtest.iloc[::,[0,2]]
###plot
plt.scatter(gbm22.predict(dt[predictors]),dt['IC50s'])
plt.plot([0,1],[0,1],'--k',transform=plt.gca().transAxes)
plt.xlabel('prediction')
plt.ylabel('True value')
plt.title('GradientBoostingResgressor')
#############randomforest puning
####determinate n_estimators   
####best {'n_estimators': 120}, 
para_test1={'n_estimators':np.arange(100,201,10)}
gsearch1=GridSearchCV(estimator=RandomForestRegressor(min_samples_split=5,min_samples_leaf=1,max_depth=3,max_features='sqrt'),param_grid=para_test1,cv=5,scoring='neg_mean_squared_error')
gsearch1.fit(dt[predictors],dt['IC50s'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
###determinate max_depth and min_sample_split, min_sample_leaf, max_feature
###best {'max_depth': 4, 'min_samples_split': 7}, 
para_test2={'max_depth':np.arange(2,10,1),'min_samples_split':np.arange(3,10,1)}
gsearch2=GridSearchCV(estimator=RandomForestRegressor(n_estimators=120,max_features='sqrt'),param_grid=para_test2,cv=5,scoring='neg_mean_squared_error')
gsearch2.fit(dt[predictors],dt['IC50s'])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#
para_test2={'max_depth':np.arange(9,20,1)}
gsearch2=GridSearchCV(estimator=RandomForestRegressor(n_estimators=120,max_features='sqrt',min_impurity_split=3),param_grid=para_test2,cv=5,scoring='neg_mean_squared_error')
gsearch2.fit(dt[predictors],dt['IC50s'])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
###
para_test3={'min_samples_leaf':np.arange(1,10,1)}
gsearch3=GridSearchCV(estimator=RandomForestRegressor(n_estimators=120,max_depth=15,min_samples_split=3,max_features='sqrt'),param_grid=para_test3,cv=5,scoring='neg_mean_squared_error')
gsearch3.fit(dt[predictors],dt['IC50s'])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#####best{'min_samples_leaf': 5}, 
para_test4={'max_features':np.arange(5,30,2)}
gsearch4=GridSearchCV(estimator=RandomForestRegressor(n_estimators=120,max_depth=15,min_samples_split=3,min_samples_leaf=3),param_grid=para_test4,cv=5,scoring='neg_mean_squared_error')
gsearch4.fit(dt[predictors],dt['IC50s'])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#### best {'max_features': 19}
rf1=RandomForestRegressor(n_estimators=120,max_depth=15,min_samples_split=3,min_samples_leaf=3,max_features=13)
modelfit(rf1,dt,predictors)
rf11=rf1.fit(dt[predictors],dt['IC50s'])
ytest=rf11.predict(dtest[predictors])
dtest.iloc[::,2]=ytest
result1=dtest.iloc[::,[0,2]]
result1.to_csv("F:/xwq/XWQ/ustmafs/math6380/miniproject2/data/drug1/OneDrug_test2.csv")
####
plt.scatter(rf11.predict(dt[predictors]),dt['IC50s'])
plt.plot([0,1],[0,1],'--k',transform=plt.gca().transAxes)
plt.xlabel('prediction')
plt.ylabel('True value')
plt.title('RandomForestRegressor')