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
sys.path.insert(2, './models')

from model_feedinput_pipeline import get_dataset_paths
from autoencoder_lstm.Bvad_AutoEncoder import Bvad_AutoEncoder
from autoencoder_lstm.ScalerWrapper import ScalerWrapper
from model_feedinput_pipeline import CODE_ENV, DATASET_ID
from autoencoder_lstm.autoencoder_lstm_generate_features import get_time_features
import autoencoder_lstm.autoencoder_lstm_main as ae_lstm

#######################################################################
# Setup Environment
#######################################################################
#Setup greengrasscore ipc
TIMEOUT = 10
flag_enable_mqtt=False

#Setup Running environment
comp_ver = "1.0.0"
code_env = CODE_ENV.WSL
sys_dataset_id = 2
select_input_stepsize = 200
if len(sys.argv) > 1:
    comp_ver = sys.argv[1]
    
    sys_code_env = sys.argv[2]
    print("'",sys_code_env,"'")
    str_to_codeenv_map={"EC2":CODE_ENV.EC2,"DEV":CODE_ENV.DEV,"WSL":CODE_ENV.WSL}
    code_env = str_to_codeenv_map[sys_code_env]

    sys_dataset_id = sys.argv[3]
    curr_dataset = int(sys_dataset_id)

    sys_dataset_stepsize = int(sys.argv[4])
    select_input_stepsize = sys_dataset_stepsize

    print(comp_ver, code_env, curr_dataset, select_input_stepsize)

#Setup Data Source
recorded_thresholds = [
        2.725298151678263
        , 2.677191266177443
        , 1.1472609170475838
        , 2.2959081453652157
]

#Model is pre-fixed to 2nd Dataset
training_set = 0
restored_model = tf.keras.models.load_model('bvad_ae_lstm_'+str(training_set))

df_features = pd.read_csv(ae_lstm.time_feature_data_filename[curr_dataset])
df_features = df_features.set_index('filename')

scaler = ScalerWrapper()
scaler.load_scaler(ae_lstm.scaler_filenames[curr_dataset])
_, X_scaled_test = scaler.transform(df_features)

auto_encoder = Bvad_AutoEncoder(test=df_features, scaler=scaler, model=restored_model)
auto_encoder.pred_autoencoder(threshold=recorded_thresholds[training_set])

X_test_pred, pred_threshold, test_scored = auto_encoder.get_test_result()
predict_results={
    'test_pred':X_test_pred
    , 'threshold':pred_threshold
    , 'score':test_scored
}

if flag_enable_mqtt:
    print('#'*80)
    print('Begin')
    ipc_client = awsiot.greengrasscoreipc.connect()
    topic = "$aws/rules/bvad_status_telemetry"
    message = "Reading Value = "
    clientType = "Mimic Bearing Anomaly Sensor"
    qos = QOS.AT_LEAST_ONCE
    print('-'*80)

df_feature = pd.DataFrame() 

#for fileindex in range(0,len(dataset_details[curr_dataset]['paths']),select_input_stepsize):
for indx in range(0, len(df_features.index), select_input_stepsize):
    transmit_data_dict={
            'dataset'  : curr_dataset + 1,
            'filename' : str(df_features.index[indx]),
            'Loss_MAE' : predict_results['score']['Loss_mae'].iloc[indx],
            'Threshold': predict_results['score']['Threshold'].iloc[indx],
            'Anomaly'  : int(predict_results['score']['Anomaly'].iloc[indx])
        }
    #print(transmit_data_dict)
        
    #interactive reporting of progress
    if indx % 100 == 0:
        print(json.dumps(transmit_data_dict))
        print('Processed ', indx, ' out of ', df_features.shape[0])

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
