from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC

# 1) Fit a logistic regression model to the mushrooms dataset from the UCI data repository using 10-fold cross-validation.

from ctypes import util
import numpy as np
from sklearn import linear_model, metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import util
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

print(sorted(sklearn.metrics.SCORERS.keys()))

#datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna.csv'
datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna_10pct.csv'

X = pd.read_csv(datafile)

import time
start_time = time.time()

le = LabelEncoder()

Labels = util.get_labels()

y = X.pop('Label')
y = le.fit_transform(y)

# We do cross val and also try holdout method!
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

svc = LinearSVC()

# BNR train the model
svc.fit(X_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

#PLOT ROC Curve 
#metrics.plot_roc_curve(svc, X_test, y_test) 
#plt.show()

# PLOT Confusion Matrix
plot_confusion_matrix(svc, X_test, y_test)  
plt.show()