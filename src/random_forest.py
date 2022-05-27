from tabnanny import verbose
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn import linear_model, metrics
import pandas as pd
import util
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

#datafile = '../data/new_set/merged_dropna.csv'
datafile = '../data/new_set/merged_dropna_10pct.csv'

X = pd.read_csv(datafile)
X = util.remove_nullnan(X)
X = util.drop_irrelevant_features(X)

y = X.pop('Label')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

import time
start_time = time.time()

clf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1)
clf.fit(X, y)

print("--- %s seconds ---" % (time.time() - start_time))

# save the model to disk
filename = util.get_model_dir() + 'random_forest.pkl'
pickle.dump(clf, open(filename, 'wb'))

#PLOT ROC Curve 
metrics.plot_roc_curve(clf, X_test, y_test) 
plt.show()