from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import util
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from joblib import parallel_backend

#datafile = '../data/new_set/merged_dropna.csv'
datafile = '../data/new_set/merged_dropna_10pct.csv'

print('read data')
X = pd.read_csv(datafile)

X = util.remove_nullnan(X)
X = util.drop_irrelevant_features(X)

import time
start_time = time.time()

le = LabelEncoder()
Labels = util.get_labels()

y = X.pop('Label')
y = le.fit_transform(y)


scaler = StandardScaler()
X  = scaler.fit_transform(X)

print('train model')
clf = LinearSVC(loss="hinge", C=1.0, max_iter=1000, verbose=1)
#clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
#clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, max_iter=1000, class_weight='balanced', verbose=1), n_jobs=-1)
#nonlinear = svm.NuSVC(nu=0.9, max_iter=1000)

print(clf.fit(X, y))
#print(nonlinear.fit(X, y))

print("--- %s seconds ---" % (time.time() - start_time))

# save the model to disk
filename = util.get_model_dir() + 'svm_linear_C-1_iter_1000.pkl'
pickle.dump(clf, open(filename, 'wb'))

#clf = 0
# save the model to disk
#filename = util.get_model_dir() + 'svm.pkl'
#with open(filename, 'rb') as f:
#    clf = pickle.load(f)

#print('crossval')
num_splits = 10
kfold = KFold(n_splits=num_splits)

results_kfold = cross_val_score(clf, X, y, cv=kfold)

print("KFold (10) Overall Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
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



print(results_kfold)
print("%0.2f accuracy with a standard deviation of %0.4f" % (results_kfold.mean(), results_kfold.std()))

precision = cross_val_score(clf, X, y, cv=5, scoring='precision')
print("%0.2f precision with a standard deviation of %0.4f" % (precision.mean(), precision.std()))

recall = cross_val_score(clf, X, y, cv=5, scoring='recall')
print("%0.2f recall with a standard deviation of %0.4f" % (recall.mean(), recall.std()))

