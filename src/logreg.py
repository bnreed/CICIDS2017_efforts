# Fit a logistic regression model or a linear regression model to the data source you
#  identified in Module #1. Use the AIC and/or BIC criteria, and K-Fold cross validation 
# to determine a best-fitting model. Interpret the coefficients of the model in the 
# context of your chosen dataset and write a short paragraph describing your findings.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from math import log
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn
import util
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

print(sorted(sklearn.metrics.SCORERS.keys()))

#datafile = '../data/new_set/merged_dropna.csv'
datafile = '../data/new_set/merged_dropna_10pct.csv'

X = pd.read_csv(datafile, header=0)
X = util.remove_nullnan(X)
X = util.drop_irrelevant_features(X)

# BNR: treat categorical values correctly (e.g. port number)

y = X.pop("Label")

import time
start_time = time.time()

Labels = util.get_labels()

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#scaler = StandardScaler()
#X  = scaler.fit_transform(X)

num_splits = 10

kfold = model_selection.KFold(n_splits=num_splits)

clf = LogisticRegression(class_weight="balanced").fit(X, y)

print("--- %s seconds ---" % (time.time() - start_time))

scoring = {'mean_squared_error': 'neg_mean_squared_error'}

cross_val_scores = cross_validate(clf, X, y, cv=kfold, scoring=scoring, verbose = 1)
print(cross_val_scores)

aic = 0
bic = 0
aic_tmp = 0
bic_tmp = 0 
count = 0
score = 0
score_tmp = 0

for score in cross_val_scores['test_mean_squared_error']:
    aic_tmp = aic_tmp + util.calculate_aic(len(y), cross_val_scores['test_mean_squared_error'][count], clf.coef_.shape[0])
    print('*** AIC: %.4f' % (aic_tmp / (count +1)))
    bic_tmp = bic_tmp + util.calculate_bic(len(y), cross_val_scores['test_mean_squared_error'][count], clf.coef_.shape[0])
    print('*** BIC: %.4f' % (bic_tmp/ (count + 1)))
    score_tmp = (score_tmp + cross_val_scores['test_mean_squared_error'][count]) / (count + 1)
    count = count + 1

aic = aic_tmp / count
bic = bic_tmp / count
test_mean_squared_error = score / count
print('aic: ' + str(aic))
print('bic: ' + str(bic))
print('test_mean_squared_error: ' + str(score_tmp / len(cross_val_scores['test_mean_squared_error'] )))

results_kfold = cross_val_score(clf, X, y, cv=kfold)

print("KFold (10) Overall Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
print(results_kfold)

print(results_kfold)
print("%0.2f accuracy with a standard deviation of %0.4f" % (results_kfold.mean(), results_kfold.std()))

precision = cross_val_score(clf, X, y, cv=5, scoring='precision')
print("%0.2f precision with a standard deviation of %0.4f" % (precision.mean(), precision.std()))

recall = cross_val_score(clf, X, y, cv=5, scoring='recall')
print("%0.2f recall with a standard deviation of %0.4f" % (recall.mean(), recall.std()))

scoring = {'mean_squared_error': 'neg_mean_squared_error'}
cross_val_scores = cross_validate(clf, X, y, cv=kfold, scoring=scoring, verbose = 1)
print(cross_val_scores)
