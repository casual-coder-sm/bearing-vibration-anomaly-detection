#######################################################################
# imports
#######################################################################

import os
import sys
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf

import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    QOS,
    PublishToIoTCoreRequest
)

from  sys import path as sys_path
sys_path.insert(0, '/root/models')
sys_path.insert(1, '/root/scripts_mimic_iot')

from  model_feedinput_pipeline import get_dataset_paths 
from model_feedinput_pipeline import CODE_ENV, DATASET_ID
from autoencoder_lstm.autoencoder_lstm_generate_features import get_time_feature

from sklearn.preprocessing import StandardScaler

def scale_timefeature_data(X, restored_scaler):
    # normalize the data
    scaled_X = restored_scaler.transform(X)
    return scaled_X


def prepare_lstm_input(X):
    # reshape inputs for LSTM [samples, timesteps, features]
    X_out = X.reshape(X.shape[0], 1, X.shape[1])
    return X_out


def pred_test_autoencoder(model, X, test, threshold):        
    # calculate the loss on the test set
    X_pred = model.predict(X, verbose=0)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index

    test_scored = pd.DataFrame(index=test.index)
    XTest = X.reshape(X.shape[0], X.shape[2])
    test_scored['Loss_mae'] = np.mean(np.abs(X_pred-XTest), axis = 1)
    test_scored['Threshold'] = threshold
    test_scored['Anomaly'] = test_scored['Loss_mae'] > test_scored['Threshold']
    return test_scored, X_pred, XTest

#######################################################################
# Setup Environment
#######################################################################
#Setup greengrasscore ipc
TIMEOUT = 10
flag_enable_mqtt=True

#Setup Running environment
comp_ver = "1.0.0"
code_env = CODE_ENV.DEV
sys_dataset_id = 2
select_input_stepsize = 200
if len(sys.argv) > 1:
    comp_ver = sys.argv[1]
    
    sys_code_env = int(sys.argv[2])
    code_env = CODE_ENV(sys_code_env)

    sys_dataset_id = int(sys.argv[3])
    curr_dataset = DATASET_ID(sys_dataset_id)

    sys_dataset_stepsize = int(sys.argv[4])
    select_input_stepsize = sys_dataset_stepsize

    print(comp_ver, code_env, curr_dataset)

#Setup Data Source
dataset_details = get_dataset_paths(code_env)   
restored_model  = tf.keras.models.load_model('/root/models/bvad_ae_lstm')
restored_scaler = joblib.load('/root/models/autoencoder_scaler')

selected_columns = ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
if curr_dataset == DATASET_ID.First:
    select_columns = {
        1 : ['b1_ch1', 'b2_ch3', 'b3_ch5', 'b4_ch7'],
        2 : ['b1_ch2', 'b2_ch4', 'b3_ch6', 'b4_ch8'],
    }
    selected_columns = select_columns[2]
print('columns chosen for training = ', selected_columns)

tf_file_indx = 0
cut_off_date_time = '2003-11-20 23:54:03'
if curr_dataset == DATASET_ID.First:
    tf_file_indx = 0 #0 or 1
    cut_off_date_time = '2003-11-20 23:54:03'
elif curr_dataset == DATASET_ID.Second:
    tf_file_indx = 2
    cut_off_date_time = '2004-02-15 12:52:39'
elif curr_dataset == DATASET_ID.Third:
    tf_file_indx = 3
    cut_off_date_time = '2004-04-08 23:51:57'

if flag_enable_mqtt:
        print('#'*80)
        print('Begin')
        ipc_client = awsiot.greengrasscoreipc.connect()
        topic = "bvad/anomaly_status"
        message = "Reading Value = "
        clientType = "Mimic Bearing Anomaly Sensor"
        qos = QOS.AT_LEAST_ONCE
        print('-'*80)

df_feature = pd.DataFrame() 
#path_list = pd.Series([ Path(filepath).name for filepath in dataset_details[curr_dataset]['paths']])
#path_list = pd.to_datetime(path_list, format='%Y.%m.%d.%H.%M.%S')
#path_list_indx = path_list[path_list>cut_off_date_time].index
#print(list(path_list_indx)[::select_input_stepsize])
#for fileindex in list(path_list_indx)[::select_input_stepsize]:

for fileindex in range(0,len(dataset_details[curr_dataset]['paths']),select_input_stepsize):
    filepath = dataset_details[curr_dataset]['paths'][fileindex]

    #get time feature
    df_feature = get_time_feature(code_env, dataset_details, curr_dataset, fileindex, selected_columns)
    df_feature['filename'] = pd.to_datetime(df_feature['filename'], format='%Y.%m.%d.%H.%M.%S')
    df_feature = df_feature.rename(columns={'filename':'date_time'})
    df_feature = df_feature.set_index('date_time')

    X = scale_timefeature_data(df_feature, restored_scaler)
    X = prepare_lstm_input(X)

    threshold = 1.756770015261056
    score, XPred, XTest = pred_test_autoencoder(restored_model, X, df_feature, threshold)    

    transmit_data_dict={
            'dataset'  : curr_dataset.value + 1,
            'filename' : str(df_feature.index[0]),
            'Loss_MAE' : score['Loss_mae'].iloc[0],
            'Threshold': score['Threshold'].iloc[0],
            'Anomaly'  : int(score['Anomaly'].iloc[0])
        }
    #print(transmit_data_dict)
        
    #interactive reporting of progress
    #if fileindex % 100 == 0:
    print(json.dumps(transmit_data_dict))
    print('Processed ', fileindex, ' out of ', len(dataset_details[curr_dataset]['paths']) )

    if flag_enable_mqtt:
        #######################################################################
        #MQTT Publish
        request = PublishToIoTCoreRequest()
        request.topic_name = topic

        payload = json.dumps(transmit_data_dict)
        request.payload = bytes(payload, "utf-8")
        request.qos = qos

        operation = ipc_client.new_publish_to_iot_core()
        operation.activate(request)

        future_response = operation.get_response()
        future_response.result(TIMEOUT)
        print('#'*80)
        #######################################################################
