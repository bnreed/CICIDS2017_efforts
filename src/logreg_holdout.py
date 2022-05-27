from sklearn.linear_model import LogisticRegression

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
import pickle
import util

print(sorted(sklearn.metrics.SCORERS.keys()))

datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna.csv'
#datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna_10pct.csv'

X = pd.read_csv(datafile)
#X = util.remove_nullnan(X)

print(X.head())
X = util.remove_nullnan(X)
X = util.drop_irrelevant_features(X)

y = X.pop('Label')
#y = le.fit_transform(y)
print('********************')
print(y.shape)

# We do cross val and also try holdout method!
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

import time
start_time = time.time()

svc = LogisticRegression()
svc.fit(X_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

## save the model to disk
filename = util.get_model_dir() + 'logreg_holdout.pkl'
pickle.dump(svc, open(filename, 'wb'))


#svc = 0
# save the model to disk
#filename = util.get_model_dir() + 'logreg_holdout.pkl'
#with open(filename, 'rb') as f:
#    svc = pickle.load(f)

#metrics.plot_roc_curve(svc, X_test, y_test) 
#plt.show()



#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

y_score = svc.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=svc.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=svc.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()

predictions = svc.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Use score method to get accuracy of model
score = svc.score(X_test, y_test)
print(score)


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()