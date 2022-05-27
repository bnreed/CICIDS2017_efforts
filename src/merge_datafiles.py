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

#pd.set_option('display.max_columns', None)

X = util.assemble_dataset()

#X = util.drop_irrelevant_features(X)

import time
start_time = time.time()
# BNR dunno what I'm doing but ~50 rows contain the headers. Defecto bug fix, possible source of data corruption.
values = ["Protocol", "Dst Port"]
#drop rows that contain any value in the list
X = X[X.Protocol.isin(values) == False]

X = util.remove_nullnan(X)

print("--- %s seconds ---" % (time.time() - start_time))

outfolder = util.get_outdir()

print('writing merged_dropna.csv')
print(X.to_csv(outfolder + 'merged_dropna.csv', index=False, header=True))

X = X.sample(frac=.01)
print('writing merged_dropna_10pct.csv')
print(X.to_csv(outfolder + 'merged_dropna_10pct.csv', index=False, header=True))
