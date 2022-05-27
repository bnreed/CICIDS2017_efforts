import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import util

# https://www.kaggle.com/datasets/karenp/cse-cic-ids2018

datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna.csv'
#datafile = '/home/bnreed/projects/EMSE6575/final_project/data/new_set/merged_dropna_10pct.csv'

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


def plot_all_data(df):

    df.drop(df.index[df['Label'] == 'Benign'], inplace=True)
    X = df
    y = X.pop('Label')

    # make a plot number of labels
    sns.set(rc={'figure.figsize':(12, 6)})
    plt.xlabel('Attack Type')
    plt.xticks(rotation=45)
    sns.set_theme()
    ax = sns.countplot(x=y, data=X)
    ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
    ax.set_yscale("log", base=10)
    plt.tight_layout()
    plt.show()

def plot_benign_vs_attack(df):

    df_benign = df[df["Label"] == 'Benign']
    df_attack = df.drop(df.index[df['Label'] == 'Benign'], inplace=False)

    xlabel = ['Benign', 'Attack']
    xdata = [df_benign, df_attack]

    print('df_benign.shape')
    print(df_benign.shape)
    print('df_attack.shape')
    print(df_attack.shape)

    # make a plot number of labels
    sns.set(rc={'figure.figsize':(12, 6)})
    plt.xlabel('Attack Type')

# https://stackoverflow.com/questions/43152502/how-can-i-rotate-xticklabels-in-matplotlib-so-that-the-spacing-between-each-xtic
    plt.xticks(rotation=45, ha="right")

    sns.set_theme()
    ax = sns.countplot(x=xlabel, data=xdata, color='blue')
    ax.set(xlabel='Attack Type', ylabel='Number of Attack Datapoints')
    plt.margins( x=200, y=200)
    #plt.tight_layout()
    plt.show()

import time
start_time = time.time()
df = pd.read_csv(datafile)
print("--- %s seconds ---" % (time.time() - start_time))


print('df.shape')

print(df.shape)

plot_all_data(df)

plot_benign_vs_attack(df)





