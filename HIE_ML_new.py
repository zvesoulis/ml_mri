# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 08:08:06 2023

@author: Zach

This code take an input CSV file containing human scored MRI data using the Trivedi 2017 system and several clinical data points.  Several different ML and regression models are trained using 5-fold cross validation and multiple metrics reported.

This is the underlying code for the manuscript--
Title: Deep learning to optimize MRI prediction of motor outcomes after HIE
Authors: Vesoulis ZA, Trivedi SB, Morris HF, McKinstry RC, Mathur AM, Wu Y

Final citation pending

%Copyright (c) 2023 Washington University 
%Created by: Zachary Vesoulis
%By using this software, you indicate agreement with the terms and
%conditions outlined in COPYRIGHT.TXT

"""

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, recall_score, make_scorer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

#load data
dataset = pd.read_csv('mri-data3.csv')

# calling head() method   
# storing in new variable  
data_top = dataset.head()  

# initialize an empty list to store the accuracies for each fold
accuracies = []
accuracies_cat =[]
accuracies_top =[]
accuracies_step=[]

recalls = []
recalls_cat = []
recalls_top = []
recalls_step=[]

# initialize empty list to store SHAP values
shap_values = []
shap_values_cat =[]
shap_values_step =[]
    
# display  
data_top 


####### COMPREHENSIVE MODEL

# split data into X and y
X = dataset.iloc[:,2:83]
y = dataset.iloc[:,83]


# create DMatrix for XGB
dtrains = xgb.DMatrix(X, label=y)

# set up the cross-validation
seed = 10
n_folds = 5
kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

# initialize the model
xgb_model = xgb.XGBRegressor(learning_rate=0.01, max_depth=5)

# run cross-validation to calculate MSE score
mse_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    xgb_model.fit(X_train, y_train)
    mse_scores.append(mean_squared_error(y_test, xgb_model.predict(X_test)))
    
# calculate the average score
avg_mse_score = np.mean(mse_scores)
print('Average MSE score:', avg_mse_score)

# run cross-validation to calculate ROC AUC score
roc_auc_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring='roc_auc')
avg_roc_auc_score = np.mean(roc_auc_scores)
print('Average ROC AUC score:', avg_roc_auc_score)

# define the number of folds for cross-validation
n_splits = 5

# initialize a KFold instance for splitting the data
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# loop over the folds
for train_index, test_index in kf.split(X):
    # split the data into train and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # fit the model on the training data
    xgb_model.fit(X_train, y_train)

    # make predictions on test data
    y_pred = xgb_model.predict(X_test)

    # round predictions to the nearest integer
    predictions = [round(value) for value in y_pred]

    # calculate accuracy of predictions for this fold
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracies.append(accuracy)
    recalls.append(recall)

    # calculate SHAP values for this fold
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_fold = explainer(X_test)
    shap_values.append(shap_values_fold.values)

# calculate and print the average accuracy over all folds
average_accuracy = sum(accuracies) / len(accuracies)
print("Average accuracy for full model: %.2f%%" % (average_accuracy * 100.0))

# calculate and print the average recall over all folds
average_recall = sum(recalls) / len(recalls)
print("Average recall for full model: %.2f%%" % (average_recall * 100.0))

# create a summary plot for SHAP values
shap_values = np.concatenate(shap_values, axis=0)

fig1=plt.figure()
shap.summary_plot(shap_values, X, max_display=10, cmap = "plasma")
fig1.savefig('shap_summary_plot.pdf', dpi=700)
plt.show()

# compute SHAP values
explainer = shap.Explainer(xgb_model, X)
shap_values2 = explainer(X)

# create a summary bar plot for SHAP values
fig3=plt.figure()
shap.summary_plot(shap_values, X, max_display=10, plot_type="bar")
fig3.savefig('shap_summary_bar_plot.pdf', dpi=700)
plt.show()


####### BASELINE CATEGORICAL MRI MODEL

# split data into X and y
X_cat = dataset.iloc[:,76]
X_cat2=X_cat.values
X_cat2=X_cat2.reshape(-1,1)
y_cat = dataset.iloc[:,83]
y_cat2=y_cat.values
y_cat2=y_cat2.reshape(-1,1)


# create DMatrix for XGB
dtrains_cat = xgb.DMatrix(X_cat, label=y_cat)

# set up the cross-validation
seed = 10
n_folds = 5
kf_cat = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

# initialize the model
xgb_model_cat = xgb.XGBRegressor(learning_rate=0.01, max_depth=5)

# define the number of folds for cross-validation
n_splits = 5

# initialize a KFold instance for splitting the data
kf_cat = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# loop over the folds
for train_index, test_index in kf_cat.split(X_cat):
    # split the data into train and test sets for this fold
    X_train_cat, X_test_cat = X_cat.iloc[train_index], X_cat.iloc[test_index]
    y_train_cat, y_test_cat = y_cat.iloc[train_index], y_cat.iloc[test_index]

    # fit the model on the training data
    xgb_model_cat.fit(X_train_cat, y_train_cat)

    # make predictions on test data
    y_pred_cat = xgb_model_cat.predict(X_test_cat)

    # round predictions to the nearest integer
    predictions_cat = [round(value) for value in y_pred_cat]
  

    # calculate accuracy of predictions for this fold
    accuracy_cat = accuracy_score(y_test_cat, predictions_cat)
    recall_cat = recall_score(y_test_cat, predictions_cat)
    
    accuracies_cat.append(accuracy_cat)
    recalls_cat.append(recall_cat)

    # calculate SHAP values for this fold
    explainer = shap.TreeExplainer(xgb_model_cat)
    shap_values_fold_cat = explainer(X_test_cat)
    shap_values_cat.append(shap_values_fold_cat.values)


    
# calculate and print the average accuracy over all folds
average_accuracy_cat = sum(accuracies_cat) / len(accuracies_cat)
print("Average accuracy for MRI cat model: %.2f%%" % (average_accuracy_cat * 100.0))

# calculate and print the average recall over all folds
average_recall_cat = sum(recalls_cat) / len(recalls_cat)
print("Average recall for MRI cat model: %.2f%%" % (average_recall_cat * 100.0))


# Create a logistic regression model
logreg_model = LogisticRegression()

# Define a scoring function for recall
recall_scorer = make_scorer(recall_score)

# Perform 5-fold cross-validation
cross_val_scores_accuracy = cross_val_score(logreg_model, X_cat2, y_cat2, cv=5, scoring='accuracy')
cross_val_scores_recall = cross_val_score(logreg_model, X_cat2, y_cat2, cv=5, scoring=recall_scorer)

# Print the cross-validation scores for each fold
for fold, (accuracy, recall) in enumerate(zip(cross_val_scores_accuracy, cross_val_scores_recall), start=1):
    print(f"Fold {fold}: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}")

# Print the mean and standard deviation of cross-validation scores
mean_accuracy = np.mean(cross_val_scores_accuracy)
std_accuracy = np.std(cross_val_scores_accuracy)
mean_recall = np.mean(cross_val_scores_recall)
std_recall = np.std(cross_val_scores_recall)
print(f"Mean Accuracy for LR categorical MRI: {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
print(f"Mean Recall for LR categorical MRI: {mean_recall:.4f} +/- {std_recall:.4f}")

####### STEPWISE REGRESSION MODEL

# split data into X and y
X_step = dataset.iloc[:,[11,15,17,20,22,37,65,71,80]]
#X_step2=X_step.values
#X_step2=X_step2.reshape(-1,1)
y_step = dataset.iloc[:,83]
y_step2=y_step.values
y_step2=y_step2.reshape(-1,1)

X_step.fillna(X_step.median(), inplace=True)


# create DMatrix for XGB
dtrains_step = xgb.DMatrix(X_step, label=y_step)

# set up the cross-validation
seed = 10
n_folds = 5
kf_step = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

# initialize the model
xgb_model_step = xgb.XGBRegressor(learning_rate=0.01, max_depth=5)

# define the number of folds for cross-validation
n_splits = 5

# initialize a KFold instance for splitting the data
kf_step = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# loop over the folds
for train_index, test_index in kf_step.split(X_step):
    # split the data into train and test sets for this fold
    X_train_step, X_test_step = X_step.iloc[train_index], X_step.iloc[test_index]
    y_train_step, y_test_step = y_step.iloc[train_index], y_step.iloc[test_index]

    # fit the model on the training data
    xgb_model_step.fit(X_train_step, y_train_step)

    # make predictions on test data
    y_pred_step = xgb_model_step.predict(X_test_step)

    # round predictions to the nearest integer
    predictions_step = [round(value) for value in y_pred_step]
  

    # calculate accuracy of predictions for this fold
    accuracy_step = accuracy_score(y_test_step, predictions_step)
    recall_step = recall_score(y_test_step, predictions_step)
    
    accuracies_step.append(accuracy_step)
    recalls_step.append(recall_step)

    # calculate SHAP values for this fold
    explainer = shap.TreeExplainer(xgb_model_step)
    shap_values_fold_step = explainer(X_test_step)
    shap_values_step.append(shap_values_fold_step.values)


    
# calculate and print the average accuracy over all folds
average_accuracy_step = sum(accuracies_step) / len(accuracies_step)
print("Average accuracy for MRI cat model: %.2f%%" % (average_accuracy_cat * 100.0))

# calculate and print the average recall over all folds
average_recall_step = sum(recalls_step) / len(recalls_cat)
print("Average recall for MRI cat model: %.2f%%" % (average_recall_cat * 100.0))


# Create a logistic regression model
logreg_model = LogisticRegression()

# Define a scoring function for recall
recall_scorer = make_scorer(recall_score)

# Perform 5-fold cross-validation
cross_val_scores_accuracy = cross_val_score(logreg_model, X_step, y_step2, cv=5, scoring='accuracy')
cross_val_scores_recall = cross_val_score(logreg_model, X_step, y_step2, cv=5, scoring=recall_scorer)

# Print the cross-validation scores for each fold
for fold, (accuracy, recall) in enumerate(zip(cross_val_scores_accuracy, cross_val_scores_recall), start=1):
    print(f"Fold {fold}: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}")

# Print the mean and standard deviation of cross-validation scores
mean_accuracy = np.mean(cross_val_scores_accuracy)
std_accuracy = np.std(cross_val_scores_accuracy)
mean_recall = np.mean(cross_val_scores_recall)
std_recall = np.std(cross_val_scores_recall)
print(f"Mean Accuracy for LR stepwise model: {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
print(f"Mean Recall for LR stepwise model: {mean_recall:.4f} +/- {std_recall:.4f}")



####### REDUCED FEATURE ML MODEL
g=3

# select the g features with the highest SHAP values
shap_values = pd.DataFrame(shap_values, columns=X.columns)
top_features = shap_values.abs().mean(axis=0).nlargest(g).index
X_top = X[top_features]

# initialize the second model using the g features
xgb_model_top = xgb.XGBRegressor(learning_rate=0.01, max_depth=5)

# run cross-validation to calculate MSE score for the second model
mse_scores_top = []
for train_index, test_index in kf.split(X_top):
    X_train_top, X_test_top = X_top.iloc[train_index], X_top.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    xgb_model_top.fit(X_train_top, y_train)
    mse_scores_top.append(mean_squared_error(y_test, xgb_model_top.predict(X_test_top)))

# calculate the average score
avg_mse_score_top = np.mean(mse_scores_top)
print('Average MSE score (top', g, 'features):', avg_mse_score_top)

# run cross-validation to calculate ROC AUC score for the second model
roc_auc_scores_top = cross_val_score(xgb_model_top, X_top, y, cv=kf, scoring='roc_auc')
avg_roc_auc_score_top = np.mean(roc_auc_scores_top)
print('Average ROC AUC score (top', g, 'features):', avg_roc_auc_score_top)

# initialize empty list to store SHAP values for the second model
shap_values_top = []

# loop over the folds
for train_index, test_index in kf.split(X_top):
    # split the data into train and test sets for this fold
    X_train_top, X_test_top = X_top.iloc[train_index], X_top.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # fit the second model on the training data
    xgb_model_top.fit(X_train_top, y_train)

    # calculate SHAP values for this fold and store them in the list
    explainer_top = shap.TreeExplainer(xgb_model_top)
    shap_values_fold_top = explainer_top.shap_values(X_test_top)
    shap_values_top.append(shap_values_fold_top)
    
    # make predictions on test data
    y_pred_top = xgb_model_top.predict(X_test_top)

    # round predictions to the nearest integer
    predictions_top = [round(value) for value in y_pred_top]

    # calculate accuracy of predictions for this fold
    accuracy_top = accuracy_score(y_test, predictions_top)
    accuracies_top.append(accuracy_top)
    
    recall_top = recall_score(y_test, predictions_top)
    recalls_top.append(recall)

# calculate and print the average accuracy over all folds
average_accuracy_top = sum(accuracies_top) / len(accuracies_top)
print("Average accuracy: (top", g, "features):%.2f%%" % (average_accuracy_top * 100.0))

# calculate and print the average recall over all folds
average_recall_top = sum(recalls_top) / len(recalls_top)
print("Average recall: (top", g, "features):%.2f%%" % (average_recall_top * 100.0))
