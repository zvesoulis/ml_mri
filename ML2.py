#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:58:37 2019

@author: zach
"""


from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.ensemble #for building models
from numpy import sort
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
import shap
import xgboost
import numpy as np
from sklearn.metrics import confusion_matrix
import lime
import lime.lime_tabular
import time 
import os 
import seaborn as sns 




from sklearn import svm

# load JS visualization code to notebook
shap.initjs()

#load data
dataset = pd.read_csv('mri-data.csv')

# calling head() method   
# storing in new variable  
data_top = dataset.head()  
    
# display  
data_top 

# split data into X and y
X = dataset.iloc[:,2:80]
y = dataset.iloc[:,81]

dtrains =  xgb.DMatrix(dataset.iloc[:,2:80], label=dataset.iloc[:,81])
dtrains2 =  xgb.DMatrix(dataset.iloc[:,[12,30,39,76,79]], label=dataset.iloc[:,81])

X2 = dataset.iloc[:,[12,30,39,76,79]]
y2 = dataset.iloc[:,81]

X.fillna(X.mean(), inplace=True)
X2.fillna(X2.mean(), inplace=True)

# split data into train and test sets
seed =10
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
model.feature_importances_




sk_xgb = sklearn.ensemble.GradientBoostingRegressor(learning_rate=0.01,max_depth=5,n_estimators=1000)
sk_xgb.fit(X_train, y_train)


# make predictions for test data
y_pred = sk_xgb.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("xk_xgb Accuracy: %.2f%%" % (accuracy * 100.0))

# =============================================================================
#  #=============================================================================
# thresholds = sort(model.feature_importances_)
# for thresh in thresholds:
#       # select features using threshold
#       selection = SelectFromModel(model, threshold=thresh, prefit=True)
#       select_X_train = selection.transform(X_train)
#       # train model
#       selection_model = XGBClassifier()
#       selection_model.fit(select_X_train, y_train)
#       plot_importance(selection_model)
#       #pyplot.show()
#       # eval model
#       select_X_test = selection.transform(X_test)
#       y_pred = selection_model.predict(select_X_test)
#       predictions = [round(value) for value in y_pred]
#       accuracy = accuracy_score(y_test, predictions)
#       
#       print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
# # =============================================================================
# =============================================================================



params = {
    "eta": 0.2,
    "max_depth": 5,
    "learning_rate": 0.01,
    "objective": "binary:logistic",
    "silent": 1,
    "base_score": np.mean(y_train),
    'n_estimators': 1000,
    "eval_metric": "logloss"
}


Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 5)
dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())
dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())



modelt = xgboost.train(params, dtrains, 5000)
modelt2 = xgboost.train(params, dtrains2, 5000)



explainer = shap.TreeExplainer(modelt)
shap_values = explainer.shap_values(X)

explainer2 = shap.TreeExplainer(modelt2)
shap_values2 = explainer2.shap_values(X2)

shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)
shap.force_plot(explainer.expected_value, shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values2, X2, plot_type="bar")
shap.dependence_plot("sub_cortical", shap_values, X,interaction_index=None)
shap.dependence_plot("Putamen/GP T1 sum", shap_values, X,interaction_index=None)
shap.dependence_plot("PLIC T1 sum", shap_values, X,interaction_index=None)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values2, X2)

##Prediction on validation set
#y_pred = modelt.predict(dv)
#
## Making the Confusion Matrix
#cm = confusion_matrix(yv, (y_pred>0.5))
#print('The Confusion Matrix is: ','\n', cm)
## Calculate the accuracy on test set
#predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
#print('The Accuracy on Test Set is: ', predict_accuracy_on_test_set)

modelt3 = xgboost.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)
xgbpred=modelt3.predict(dv)
accuracy = accuracy_score(yv, xgbpred)
print("XGB Reduced Model Accuracy: %.2f%%" % (accuracy * 100.0))

# fit model no training data
model2 = XGBClassifier()
model2.fit(X_train, y_train)

plot_importance(model2)
plt.show()

# make predictions for test data
y_pred = model2.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("XGB Reduced Model Accuracy: %.2f%%" % (accuracy * 100.0))

# CV model
cv_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=0)

kfold = KFold(n_splits=5, random_state=2)
results = cross_val_score(cv_model, X2, y2, cv=kfold)
print("XGB CV Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

svc = svm.SVC(kernel='linear',gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Linear SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
scores = cross_val_score(svc, X2, y2, cv=5)
print("Linear SVM CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

