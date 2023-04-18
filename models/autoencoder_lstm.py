# %%
import os
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
from numpy.random import seed

import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
sns.set(color_codes=True)

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy



import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

seed(10)
tf.random.set_seed(10)


# %%
import model_feedinput_pipeline
from model_feedinput_pipeline import CODE_ENV, DATASET_ID


# %%
# Root Mean Squared Sum
def calculate_rms(df):
    result = []
    for col in df:
        r = np.sqrt((df[col]**2).sum() / len(df[col]))
        result.append(r)
    return np.array(result)


# extract peak-to-peak features
def calculate_p2p(df):
    return np.array(df.max().abs() + df.min().abs())


# extract shannon entropy (cut signals to 500 bins)
def calculate_entropy(df):
    ent = []
    for col in df:
        ent.append(entropy(pd.cut(df[col], 500).value_counts()))
    return np.array(ent)


# extract clearence factor
def calculate_clearence(df):
    result = []
    for col in df:
        r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
        result.append(r)
    return np.array(result)


def time_features(code_env: CODE_ENV, dataset_details, id:DATASET_ID, select_columns:list):
    #time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
    
    data = pd.DataFrame()   
    for fileindex, filepath in enumerate(dataset_details[id]['paths']):
        #step1: get raw_data and associated filename
        raw_data = model_feedinput_pipeline.get_df(dataset_details, id, fileindex, code_env)
        raw_data = raw_data[select_columns]
        if code_env == CODE_ENV.EC2:
            filename = Path(filepath.key).name
        elif code_env == CODE_ENV.WIN:
            filename =  Path(filepath).name
        
        #step2 : Generate features
        mean_abs = raw_data.abs().mean().to_numpy().reshape(1,len(select_columns))
        std = raw_data.std().to_numpy().reshape(1,len(select_columns))
        skew = raw_data.skew().to_numpy().reshape(1,len(select_columns))
        kurtosis = raw_data.kurtosis().to_numpy().reshape(1,len(select_columns))
        entropy = calculate_entropy(raw_data).reshape(1,len(select_columns))
        rms = calculate_rms(raw_data).reshape(1,len(select_columns))
        max_abs=raw_data.abs().max().to_numpy().reshape(1,len(select_columns))
        p2p = calculate_p2p(raw_data).reshape(1,len(select_columns))
        crest = (max_abs/rms)
        clearence = calculate_clearence(raw_data).reshape(1,len(select_columns))
        shape = (rms/mean_abs)
        impulse = (max_abs/mean_abs)


        #step3 : save to dataframe
        df_mean_abs = pd.DataFrame(mean_abs, columns=[c+'_mean' for c in select_columns])
        df_std = pd.DataFrame(std, columns=[c+'_std'  for c in select_columns])
        df_skew = pd.DataFrame(skew, columns=[c+'_skew' for c in select_columns])
        df_kurtosis = pd.DataFrame(kurtosis, columns=[c+'_kurtosis' for c in select_columns])
        df_entropy = pd.DataFrame(entropy, columns=[c+'_entropy' for c in select_columns])
        df_rms = pd.DataFrame(rms, columns=[c+'_rms' for c in select_columns])        
        df_max = pd.DataFrame(max_abs, columns=[c+'_max' for c in select_columns])
        df_p2p = pd.DataFrame(p2p, columns=[c+'_p2p' for c in select_columns])
        df_crest = pd.DataFrame(crest, columns=[c+'_crest' for c in select_columns])
        df_clearence = pd.DataFrame(clearence, columns=[c+'_clearence' for c in select_columns])
        df_shape = pd.DataFrame(shape, columns=[c+'_shape' for c in select_columns])
        df_impulse = pd.DataFrame(impulse, columns=[c+'_impulse' for c in select_columns])
        df_filename = pd.DataFrame([filename])
        df = pd.concat([df_filename, df_mean_abs, df_std, df_skew, df_kurtosis
                           ,df_entropy, df_rms, df_max, df_p2p, df_crest
                           ,df_clearence, df_shape, df_impulse]
                           ,axis=1)
        data = pd.concat([data, df], axis=0)
        if fileindex % 10 == 0:
            print('Processed ', fileindex, ' out of ', len(dataset_details[id]['paths']) )

    data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
    data = data.sort_index()

    return data


# %%

#####################################################################################
#***************IMP: Update coding environment********************
#####################################################################################
code_env = CODE_ENV.EC2

select_input_stepsize= 3000
curr_dataset = DATASET_ID.Second
selected_columns = ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
if curr_dataset == DATASET_ID.First:
    select_columns = {
        1 : ['b1_ch1', 'b2_ch3', 'b3_ch5', 'b4_ch7'],
        2 : ['b1_ch2', 'b2_ch4', 'b3_ch6', 'b4_ch8'],
    }
    selected_columns = select_columns[1]
print('columns chosen for training = ', selected_columns)
time_feature_data_filename=['timefeatures_1st.csv', 'timefeatures_2nd.csv', 'timefeatures_3rd.csv']

# %%
dataset_paths = model_feedinput_pipeline.get_dataset_paths(code_env)
time_feature_data = time_features(code_env, dataset_paths, curr_dataset, selected_columns)
time_feature_data.to_csv(time_feature_data_filename[curr_dataset])


# %%
merged_data = pd.read_csv(time_feature_data_filename[curr_dataset])
merged_data = merged_data.rename(columns={'Unnamed: 0':'time'})
merged_data.set_index('time')
merged_data.describe()
