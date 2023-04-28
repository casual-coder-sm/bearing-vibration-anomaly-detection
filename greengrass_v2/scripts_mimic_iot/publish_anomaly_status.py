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
from autoencoder_lstm.autoencoder_lstm_generate_features import get_time_feature
import autoencoder_lstm.autoencoder_lstm_main as ae_lstm

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
select_input_stepsize = 50
select_output_stepsize = 5
if len(sys.argv) > 1:
    comp_ver = sys.argv[1]
    
    sys_code_env = sys.argv[2]
    print("'",sys_code_env,"'")
    str_to_codeenv_map={"EC2":CODE_ENV.EC2,"DEV":CODE_ENV.DEV,"WSL":CODE_ENV.WSL}
    code_env = str_to_codeenv_map[sys_code_env]

    sys_model_id = sys.argv[3]
    training_model_id = int(sys_model_id)

    sys_dataset_stepsize = int(sys.argv[4])
    select_input_stepsize = sys_dataset_stepsize

    sys_preict_stepsize = int(sys.argv[5])
    select_output_stepsize = sys_preict_stepsize

    print(comp_ver, code_env, training_model_id, select_input_stepsize, select_output_stepsize)

predict_for_dataset=int(os.environ['predict_for_dataset'])
device_name=os.environ['device_name']

#Setup Data Source
recorded_thresholds = [
        2.725298151678263
        , 2.677191266177443
        , 1.1472609170475838
        , 2.2959081453652157
]

#Step 1: Set the datasource paths
dataset_paths = get_dataset_paths(code_env)

#Step 2: Load the Reference fitted scaler
scaler = ScalerWrapper()
scaler.load_scaler(ae_lstm.scaler_filenames[predict_for_dataset])

#Step 3: Load the Reference trained Model
restored_model = tf.keras.models.load_model('bvad_ae_lstm_'+str(training_model_id))

#Step 4: Configure settings
predict_model_columns = ae_lstm.select_columns[predict_for_dataset]
predict_model_ims_dataset = ae_lstm.dataset_id_mapping[predict_for_dataset]

time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
feature_columns=['B1', 'B2', 'B3', 'B4']
columns = [c+'_'+tf for c in feature_columns for tf in time_features]

#Step 5: For each second of data mimic generation of sensor vibration data and prection of anomaly status
for record_indx in range(0, len(dataset_paths[predict_model_ims_dataset]['paths']), select_input_stepsize):
    #Step 5.1 : Generate feature data
    df_features = pd.DataFrame()
    for i in range(record_indx, record_indx+select_output_stepsize):
        data = get_time_feature(code_env, dataset_paths, predict_model_ims_dataset, i, predict_model_columns, feature_columns)
        df_features=pd.concat([df_features, data], axis=0)
    df_features['filename'] = pd.to_datetime(df_features['filename'], format='%Y.%m.%d.%H.%M.%S')
    df_features = df_features.set_index('filename')
    df_features = df_features[columns]

    #Step 5.2 : Normalize the features
    _, X_scaled_test = scaler.transform(df_features)

    #Step 5.3: Perform Prediction
    auto_encoder = Bvad_AutoEncoder(test=df_features, scaler=scaler, model=restored_model)
    auto_encoder.pred_autoencoder(threshold=recorded_thresholds[training_model_id])
    X_test_pred, pred_threshold, test_scored = auto_encoder.get_test_result()
    predict_results={
        'test_pred':X_test_pred
        , 'threshold':pred_threshold
        , 'score':test_scored
    }

    #Step 5.4: Mimic IoT device sending Device Status
    if flag_enable_mqtt:
        print('#'*80)
        print('Begin')
        ipc_client = awsiot.greengrasscoreipc.connect()
        topic = "bvad/anomaly_status/"+device_name
        message = "Reading Value = "
        clientType = "Mimic Bearing Anomaly Sensor"
        qos = QOS.AT_LEAST_ONCE
        print('-'*80)

    transmission_data = []
    for i in range(select_output_stepsize):
        transmit_data_dict={
            'devicename': device_name,
            'dataset'   : predict_for_dataset + 1,
            'filename'  : str(df_features.index[i]),
            'Loss_MAE'  : predict_results['score']['Loss_mae'].iloc[i],
            'Threshold' : predict_results['score']['Threshold'].iloc[i],
            'Anomaly'   : int(predict_results['score']['Anomaly'].iloc[i])
        }
        transmission_data.append(transmit_data_dict)
    transmission_data_json = json.dumps(transmission_data)
    #interactive reporting of progress
    print('Processed ', record_indx, ' out of ', len(dataset_paths[predict_model_ims_dataset]['paths']))

    if flag_enable_mqtt:
        #######################################################################
        #MQTT Publish
        request = PublishToIoTCoreRequest()
        request.topic_name = topic

        payload = transmission_data_json
        request.payload = bytes(payload, "utf-8")
        request.qos = qos

        operation = ipc_client.new_publish_to_iot_core()
        operation.activate(request)

        future_response = operation.get_response()
        future_response.result(TIMEOUT)
        print('#'*80)
        #######################################################################
