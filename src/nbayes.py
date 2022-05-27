# 1) Fit a logistic regression model to the mushrooms dataset from the UCI data repository using 10-fold cross-validation.

from ctypes import util
import numpy as np
from sklearn import linear_model, metrics
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from joblib import parallel_backend


import util

print(sorted(sklearn.metrics.SCORERS.keys()))

datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna.csv'
#datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna_10pct.csv'

import time
start_time = time.time()
X = pd.read_csv(datafile)

X  = util.remove_nullnan(X)
X = util.drop_irrelevant_features(X)

le = LabelEncoder()

Labels = util.get_labels()

y = X.pop('Label')
y = le.fit_transform(y)

num_splits = 10
kfold = KFold(n_splits=num_splits)
nb = BernoulliNB()

# BNR train the model
#with parallel_backend('threading', n_jobs=12):
nb.fit(X, y)

print("--- %s seconds ---" % (time.time() - start_time))

# save the model to disk
filename = util.get_model_dir() + 'nbayes.pkl'
pickle.dump(nb, open(filename, 'wb'))

results_kfold = cross_val_score(nb, X, y, cv=kfold)

print("KFold (10) Overall Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 

print(results_kfold)
print("%0.2f accuracy with a standard deviation of %0.4f" % (results_kfold.mean(), results_kfold.std()))

precision = cross_val_score(nb, X, y, cv=5, scoring='precision')
print("%0.2f precision with a standard deviation of %0.4f" % (precision.mean(), precision.std()))

recall = cross_val_score(nb, X, y, cv=5, scoring='recall')
print("%0.2f recall with a standard deviation of %0.4f" % (recall.mean(), recall.std()))

scoring = {'mean_squared_error': 'neg_mean_squared_error'}
cross_val_scores = cross_validate(nb, X, y, cv=kfold, scoring=scoring, verbose = 1)
print(cross_val_scores)

aic = 0
bic = 0
aic_tmp = 0
bic_tmp = 0 
count = 0
score = 0
score_tmp = 0

for score in cross_val_scores['test_mean_squared_error']:
    aic_tmp = aic_tmp + util.calculate_aic(len(y), cross_val_scores['test_mean_squared_error'][count], nb.coef_.shape[0])
    print('*** AIC: %.4f' % (aic_tmp / (count +1)))
    bic_tmp = bic_tmp + util.calculate_bic(len(y), cross_val_scores['test_mean_squared_error'][count], nb.coef_.shape[0])
    print('*** BIC: %.4f' % (bic_tmp/ (count + 1)))
    score_tmp = (score_tmp + cross_val_scores['test_mean_squared_error'][count]) / (count + 1)
    count = count + 1

aic = aic_tmp / count
bic = bic_tmp / count
test_mean_squared_error = score / count
print('aic: ' + str(aic))
print('bic: ' + str(bic))
print('test_mean_squared_error: ' + str(score_tmp / len(cross_val_scores['test_mean_squared_error'] )))

results_kfold = cross_val_score(nb, X, y, cv=kfold)

print("KFold (10) Overall Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
print(results_kfold)

