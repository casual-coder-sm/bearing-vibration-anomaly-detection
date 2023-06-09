import os
import sys
from sys import path as sys_path
# Code to re-configure current working directory (for ipynb environment)
curr_dir = os.getcwd()
#check in the order sub-directory to main-directory
if 'autoencoder_lstm' in  curr_dir:
    os.chdir('..')
if 'models' in curr_dir in curr_dir:
    pass
elif 'bearing-vibration-anomaly-detection' in curr_dir:
    os.chdir('./models')
curr_dir = os.getcwd()
sys_path.insert(0, curr_dir)

from pathlib import Path

import pandas as pd
import numpy as np
from numpy.random import seed

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)

from scipy.stats import entropy

import model_feedinput_pipeline
from model_feedinput_pipeline import CODE_ENV, DATASET_ID


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

def get_time_feature(code_env: CODE_ENV, dataset_details, id:DATASET_ID, fileindex:int, select_columns:list, feature_columns:list):
    filepath = dataset_details[id]['paths'][fileindex]
    
    #step1: get raw_data and associated filename
    raw_data = model_feedinput_pipeline.get_df(dataset_details, id, fileindex, code_env)
    raw_data = raw_data[select_columns]
    mapping = {
        raw_data.columns[0]:feature_columns[0]
        , raw_data.columns[1]:feature_columns[1]
        , raw_data.columns[2]:feature_columns[2]
        , raw_data.columns[3]:feature_columns[3]
    }
    raw_data = raw_data.rename(columns=mapping)
    
    if code_env == CODE_ENV.EC2:
        filename = Path(filepath.key).name
    elif code_env == CODE_ENV.WSL or code_env == CODE_ENV.DEV:
        filename =  Path(filepath).name
    
    #step2 : Generate features
    mean_abs = raw_data.abs().mean().to_numpy().reshape(1,len(feature_columns))
    std = raw_data.std().to_numpy().reshape(1,len(feature_columns))
    skew = raw_data.skew().to_numpy().reshape(1,len(feature_columns))
    kurtosis = raw_data.kurtosis().to_numpy().reshape(1,len(feature_columns))
    entropy = calculate_entropy(raw_data).reshape(1,len(feature_columns))
    rms = calculate_rms(raw_data).reshape(1,len(feature_columns))
    max_abs=raw_data.abs().max().to_numpy().reshape(1,len(feature_columns))
    p2p = calculate_p2p(raw_data).reshape(1,len(feature_columns))
    crest = (max_abs/rms)
    clearence = calculate_clearence(raw_data).reshape(1,len(feature_columns))
    shape = (rms/mean_abs)
    impulse = (max_abs/mean_abs)

    #step3 : save to dataframe
    df_mean_abs = pd.DataFrame(mean_abs, columns=[c+'_mean' for c in feature_columns])
    df_std = pd.DataFrame(std, columns=[c+'_std'  for c in feature_columns])
    df_skew = pd.DataFrame(skew, columns=[c+'_skew' for c in feature_columns])
    df_kurtosis = pd.DataFrame(kurtosis, columns=[c+'_kurtosis' for c in feature_columns])
    df_entropy = pd.DataFrame(entropy, columns=[c+'_entropy' for c in feature_columns])
    df_rms = pd.DataFrame(rms, columns=[c+'_rms' for c in feature_columns])        
    df_max = pd.DataFrame(max_abs, columns=[c+'_max' for c in feature_columns])
    df_p2p = pd.DataFrame(p2p, columns=[c+'_p2p' for c in feature_columns])
    df_crest = pd.DataFrame(crest, columns=[c+'_crest' for c in feature_columns])
    df_clearence = pd.DataFrame(clearence, columns=[c+'_clearence' for c in feature_columns])
    df_shape = pd.DataFrame(shape, columns=[c+'_shape' for c in feature_columns])
    df_impulse = pd.DataFrame(impulse, columns=[c+'_impulse' for c in feature_columns])
    df_filename = pd.DataFrame([filename], columns=['filename'])
    df = pd.concat([df_filename, df_mean_abs, df_std, df_skew, df_kurtosis
                        ,df_entropy, df_rms, df_max, df_p2p, df_crest
                        ,df_clearence, df_shape, df_impulse]
                        ,axis=1)
    return df


def get_time_features(code_env: CODE_ENV, dataset_details, id:DATASET_ID, select_columns:list):
    data = pd.DataFrame()   
    for fileindex, filepath in enumerate(dataset_details[id]['paths']):
        #get time feature
        feature_columns=['B1', 'B2', 'B3', 'B4']
        df = get_time_feature(code_env, dataset_details, id, fileindex, select_columns, feature_columns)
        #concat with previous set
        data = pd.concat([data, df], axis=0)
        
        #interactive reporting of progress
        if fileindex % 10 == 0:
            print('Processed ', fileindex, ' out of ', len(dataset_details[id]['paths']) )
    
    data['filename'] = pd.to_datetime(data['filename'], format='%Y.%m.%d.%H.%M.%S')
    data = data.set_index('filename')

    time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
    feature_columns=['B1', 'B2', 'B3', 'B4']
    
    columns = [c+'_'+tf for c in feature_columns for tf in time_features]
    data = data[columns]
     
    return data


if __name__ == "__main__":

    #####################################################################################
    #***************IMP: Update coding environment********************
    #####################################################################################
    code_env = CODE_ENV.WSL    
    curr_dataset = DATASET_ID.Second

    comp_ver = ''
    if len(sys.argv) > 1:
        comp_ver = sys.argv[1]
        
        sys_code_env = int(sys.argv[2])
        code_env = CODE_ENV(sys_code_env)

        sys_dataset_id = int(sys.argv[3])
        curr_dataset = DATASET_ID(sys_dataset_id)

        print(comp_ver, code_env, curr_dataset)
    else:
        #For Win and EC2
        pass

    #Step 1 : Setup Data Source
    dataset_paths = model_feedinput_pipeline.get_dataset_paths(code_env)

    #Step 2 : Generate time features for the specified dataset and columns
    select_input_stepsize= 3000
    selected_columns = ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
    if curr_dataset == DATASET_ID.First:
        select_columns = {
            1 : ['b1_ch1', 'b2_ch3', 'b3_ch5', 'b4_ch7'],
            2 : ['b1_ch2', 'b2_ch4', 'b3_ch6', 'b4_ch8'],
        }
        selected_columns = select_columns[2]
    print('columns chosen for training = ', selected_columns)
    time_feature_data = get_time_features(code_env, dataset_paths, curr_dataset, selected_columns)

    #Step 3 : Save Output
    time_feature_data_filename=['timefeatures_1st.csv', 'timefeatures_2nd.csv', 'timefeatures_3rd.csv']
    time_feature_data.to_csv(time_feature_data_filename[curr_dataset.value])
    merged_data = pd.read_csv(time_feature_data_filename[curr_dataset.value])

    


