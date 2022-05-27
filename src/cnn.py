# Code from
# https://www.kaggle.com/code/azazurrehmanbutt/cicids-ids-2018-using-cnn

# import libraries
import itertools, math, os, re, time, tqdm
import keras
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.offline as pyo
import seaborn as sns

import util

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample

# BNR: for matplotlib from ssh
os.environ['DISPLAY'] = 'localhost:10.0'
import matplotlib
matplotlib.use('tkagg')

input_dir = util.get_ml_basedir()

#datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna.csv'
datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna_10pct.csv'

print('read data')
X = pd.read_csv(datafile)
X = util.drop_irrelevant_features(X)
X = util.remove_nullnan(X)

# check the available data
for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# make a plot number of labels
sns.set(rc={'figure.figsize':(12, 6)})
plt.xlabel('Attack Type')
plt.xticks(rotation=45, ha="right")
sns.set_theme()
ax = sns.countplot(x='Label', data=X)
ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
plt.show()

label_encoder = LabelEncoder()
X['Label']= label_encoder.fit_transform(X['Label'])
#print(X['Label'].unique())

Xdata = []
ydata = []
idx = 0
train_dataset = pd.DataFrame()

# make data equally representative across classes
# BNR this is wrong!!!
for label in X['Label'].unique():

    Xdf = X[X['Label'] == idx]
    Xdf = resample(Xdf, n_samples=200000, random_state=123, replace=True)
    Xdata.append(Xdf)

    idx = idx + 1

    train_dataset = pd.concat([train_dataset, Xdf], ignore_index=True)

#print(train_dataset.shape)
#print(ydata)
# viewing the distribution of intrusion attacks in our dataset 
plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.title('Intrusion Attack Type Distribution')
plt.pie(train_dataset['Label'].value_counts(), labels=['Benign', 'BF', 'BF-SSH'], colors=['blue', 'magenta', 'cyan'])
p = plt.gcf()
p.gca().add_artist(circle)