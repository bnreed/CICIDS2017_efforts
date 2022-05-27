from socket import AddressFamily
import numpy as np
import pandas as pd
from math import log

# 
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_rows',1000)

def get_ids_datatypes(): 

    ids_datatypes = {
        'Protocol': np.int8,
        'Flow Duration': np.int64,
        'Tot Fwd Pkts': np.int16,
        'Tot Bwd Pkts': np.float64,
        'TotLen Fwd Pkts': np.float64,
        'TotLen Bwd Pkts': np.int32,
        'Fwd Pkt Len Max': np.int32,
        'Fwd Pkt Len Min': np.float64,
        'Fwd Pkt Len Mean': np.float64,
        'Fwd Pkt Len Std': np.float64,
        'Bwd Pkt Len Max': np.int16,
        'Bwd Pkt Len Min': np.float64,
        'Bwd Pkt Len Mean': np.float64,
        'Bwd Pkt Len Std': np.float64,
        'Flow Byts/s': np.float64,
        'Flow Pkts/s': np.float64,
        'Flow IAT Mean': np.float64,
        'Flow IAT Std': np.float64,
        'Flow IAT Max': np.int64,
        'Flow IAT Min': np.int32,
        'Fwd IAT Tot': np.float64,
        'Fwd IAT Mean': np.float32,
        'Fwd IAT Std': np.float64,
        'Fwd IAT Max': np.int32, 
        'Fwd IAT Min': np.int32,
        'Bwd IAT Tot': np.float64,
        'Bwd IAT Mean': np.float64,
        'Bwd IAT Std': np.float64,
        'Bwd IAT Max': np.int64,
        'Bwd IAT Min': np.int64,
        'Fwd PSH Flags': np.int8,
        'Bwd PSH Flags': np.int8,
        'Fwd URG Flags': np.int8,
        'Bwd URG Flags': np.int8,
        'Fwd Header Len': np.int32,
        'Bwd Header Len': np.float64,
        'Fwd Pkts/s' : np.float64,
        'Bwd Pkts/s': np.float64,
        'Pkt Len Min': np.int16,
        'Pkt Len Max': np.float64,
        'Pkt Len Mean': np.float64,
        'Pkt Len Std': np.float64,
        'Pkt Len Var': np.float64,
        'FIN Flag Cnt': np.int8,
        'SYN Flag Cnt': np.int8,
        'RST Flag Cnt': np.int8,
        'PSH Flag Cnt': np.int8,
        'ACK Flag Cnt': np.int8,
        'URG Flag Cnt': np.int8,
        'CWE Flag Count': np.int8,
        'ECE Flag Cnt': np.int8,
        'Pkt Size Avg': np.float32,
        'Fwd Seg Size Avg': np.float32,
        'Bwd Seg Size Avg': np.float32,
        'Fwd Byts/b Avg': np.int8,
        'Fwd Pkts/b Avg': np.int8,
        'Fwd Blk Rate Avg': np.int8,
        'Bwd Byts/b Avg': np.int8,
        'Bwd Pkts/b Avg': np.int8,
        'Bwd Blk Rate Avg': np.int8,
        'Subflow Fwd Pkts': np.int16,
        'Subflow Fwd Byts': np.int32,
        'Subflow Bwd Pkts': np.int16,
        'Subflow Bwd Byts': np.int32,
        'Init Fwd Win Byts': np.int32, 
        'Init Bwd Win Byts': np.int32,
        'Fwd Act Data Pkts': np.int16,
        'Fwd Seg Size Min': np.float64,
        'Active Mean': np.float64,
        'Active Std': np.float64,
        'Active Max': np.int32,
        'Active Min': np.float64,
        'Idle Mean': np.float64,
        'Idle Std': np.float64,
        'Idle Max': np.int64,
        'Idle Min': np.int64,
        'Label': object
    }
    return ids_datatypes

def get_ml_basedir():
    return '../data/processed/'


def get_outdir():
    return '../data/new_set/'

def get_labels():
    return ['Benign', 'Bot', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Hulk', 
          'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection',  'DDoS attacks-LOIC-HTTP',
          'Infilteration', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'FTP-BruteForce',
          'SSH-Bruteforce', 'DDOS attack-LOIC-UDP', 'DDOS attack-HOIC']

def get_files():
    return ['Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
            'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
            'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 
            'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv', 
            'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
            'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
            'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
            'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
            'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
            'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv']

def get_all_X_cols(): 
     return ['Dst Port','Protocol','Timestamp','Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts',
            'TotLen Fwd Pkts','TotLen Bwd Pkts','Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd Pkt Len Mean',
            'Fwd Pkt Len Std','Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd Pkt Len Mean','Bwd Pkt Len Std',
            'Flow Byts/s','Flow Pkts/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
            'Fwd IAT Tot','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Tot',
            'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags',
            'Fwd URG Flags','Bwd URG Flags','Fwd Header Len','Bwd Header Len','Fwd Pkts/s','Bwd Pkts/s',
            'Pkt Len Min','Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var','FIN Flag Cnt','SYN Flag Cnt',
            'RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt','CWE Flag Count','ECE Flag Cnt',
            'Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg','Bwd Seg Size Avg','Fwd Byts/b Avg',
            'Fwd Pkts/b Avg','Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg',
            'Subflow Fwd Pkts','Subflow Fwd Byts','Subflow Bwd Pkts','Subflow Bwd Byts','Init Fwd Win Byts',
            'Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean','Active Std','Active Max',
            'Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']

def get_all_cols(): 
     return ['Dst Port','Protocol','Timestamp','Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts',
            'TotLen Fwd Pkts','TotLen Bwd Pkts','Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd Pkt Len Mean',
            'Fwd Pkt Len Std','Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd Pkt Len Mean','Bwd Pkt Len Std',
            'Flow Byts/s','Flow Pkts/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
            'Fwd IAT Tot','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Tot',
            'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags',
            'Fwd URG Flags','Bwd URG Flags','Fwd Header Len','Bwd Header Len','Fwd Pkts/s','Bwd Pkts/s',
            'Pkt Len Min','Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var','FIN Flag Cnt','SYN Flag Cnt',
            'RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt','CWE Flag Count','ECE Flag Cnt',
            'Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg','Bwd Seg Size Avg','Fwd Byts/b Avg',
            'Fwd Pkts/b Avg','Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg',
            'Subflow Fwd Pkts','Subflow Fwd Byts','Subflow Bwd Pkts','Subflow Bwd Byts','Init Fwd Win Byts',
            'Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean','Active Std','Active Max',
            'Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label']



def get_data_shapes():

    base_dir = get_ml_basedir()

    data = pd.DataFrame()
  
    for file in get_files():

        print(base_dir + file)
        
        df = pd.read_csv(base_dir + file,  header = 0)
        print('df.shape ' + file)
        print(df.shape)
        print(df.head)

def assemble_dataset():

    #ids_datatypes = get_ids_datatypes()
    #used_cols = (ids_datatypes.keys())
    #datatypes = ids_datatypes

    base_dir = get_ml_basedir()

    data = pd.DataFrame()

    for file in get_files():

        print(base_dir + file)
        df = pd.read_csv(base_dir + file, header=0, names=get_all_cols())
        print('df.shape')
        print(df.shape)
        #print(df.head)
        data = pd.concat([data, df], ignore_index=True)

    print('data.shape')
    print(data.shape)

    return data

def remove_nullnan(X):
    X = X.dropna()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    return X


def drop_irrelevant_features(df):
    df = df.drop('Timestamp', axis = 1)
    #df = df.drop('Dst IP', axis = 1)
    #df = df.drop('Src IP', axis = 1)
    #df = df.drop('Src Port', axis = 1)
    #df = df.drop('Flow ID', axis = 1)
    return df

# calculate aic for regression 
# Score = kp - (2 * log (L))
# k = 2 ; L = mse; p = num parameters

def calculate_aic(n, mse, num_params):
    aic = (2 * num_params) - (2 * log(abs(mse)))
    return aic

# calculate bic for regression ..... (log(n) * k) - (2 * log(L))
# Score = kp - (2 * log (L)
# k = log(sample size); L = mse; p = num parameters; 

def calculate_bic(n, mse, num_params):
    bic =  (log(n) * num_params)  - (2 * log(abs(mse)))
    return bic

def get_model_dir():
    return '../models/'