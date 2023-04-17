#######################################################################
# imports
#######################################################################
import numpy as np
import pandas as pd

import os
import time
import datetime

from pathlib import Path

import json

import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    QOS,
    PublishToIoTCoreRequest
)
import time

#######################################################################
# Setup Environment
#######################################################################
#part 1: Setup greengrasscore ipc
TIMEOUT = 10

print('#'*80)
ipc_client = awsiot.greengrasscoreipc.connect()
topic = "collect_device_data/device_reading"
message = "Reading Value = "
clientType = "Mimic Bearing Anomaly Sensor"
qos = QOS.AT_LEAST_ONCE

#part 2: Setup bearing vibration input data configurations
parent_folder_raw_data='/root/datasets/phm-ims-datasets'
data_set1_path = parent_folder_raw_data + '/1st_test'
data_set2_path = parent_folder_raw_data + '/2nd_test'
data_set3_path = parent_folder_raw_data + '/3rd_test'

#1st Set has 8 
col_names_1st = ['b1_ch1', 'b1_ch2', 'b2_ch3', 'b2_ch4', 'b3_ch5', 'b3_ch6', 'b4_ch7', 'b4_ch8']
#2nd and 3rd has 4
col_names_2nd_3rd = ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']

data_set_paths= [data_set1_path, data_set2_path, data_set3_path]

# iterate through each filename which is date that samples were taken
# file_names should be sorted from earliest to latest for each data_set
file_names_list = [sorted(os.listdir(src_path)) for src_path in data_set_paths]

col_names_set = [col_names_1st, col_names_2nd_3rd, col_names_2nd_3rd]
select_columns = [
        [
            ['b1_ch1', 'b1_ch2'],
            ['b2_ch3', 'b2_ch4'],
            ['b3_ch5', 'b3_ch6'],
            ['b4_ch7', 'b4_ch8']
        ],
        [
            ['b1_ch1'],
            ['b2_ch2'],
            ['b3_ch3'],
            ['b4_ch4'],
        ],
        [
            ['b1_ch1'],
            ['b2_ch2'],
            ['b3_ch3'],
            ['b4_ch4'],    
        ],
    ]
print('-'*80)
for i, data_set_path in enumerate(data_set_paths):
    print(data_set_path.split(sep='/')[-1],'contains ', '{:4d}'.format(len(file_names_list[i])), ' number of files'
          '; Associated Columns =', col_names_set[i])

#######################################################################
#Test Configuration
#######################################################################
select_data_set = 0 
select_bearing = 0
file_index = 0

print(data_set_paths[select_data_set].split(sep='/')[-1],'contains ', '{:4d}'.format(len(file_names_list[select_data_set])), ' number of files')
print('Associated Columns =', col_names_set[select_data_set])
print('Selected for transmission =', select_columns[select_data_set][select_bearing])
print('Selected file name =', file_names_list[select_data_set][file_index])
print('-'*80)
#######################################################################
#Helper functions
#######################################################################
def get_bearing_data(src_path, filename, col_names,  bearing_channels):
    df_out = pd.DataFrame()        
    
    # open the file containing the measurements
    df = pd.read_csv(Path(src_path) / filename, sep="\t", names=col_names)
    
    #filter dataframe to select only required column
    df = df.loc[:,bearing_channels]
    
    return df
    

for file_index, filename in enumerate(file_names_list[select_data_set][::3000]):
    print('Begin Execution for iteration:',file_index, filename)
    df = get_bearing_data(data_set_paths[select_data_set],
                    filename,
                    col_names_set[select_data_set],
                    select_columns[select_data_set][select_bearing])

    transmit_data=[];
    for ch in select_columns[select_data_set][select_bearing]:
        values = df[ch].to_list()

        transmit_data_dict1={
            'dataset':'ims_'+ data_set_paths[select_data_set].split(sep='/')[-1],
            'filename':filename,
            'filepart':1,
            'bearing':ch,
            'readings':values[0:10240]
            }
        transmit_data_json1=json.dumps(transmit_data_dict1)
        request1 = PublishToIoTCoreRequest()
        request1.topic_name = topic
        payload = transmit_data_json1
        request1.payload = bytes(payload, "utf-8")
        request1.qos = qos
        operation1 = ipc_client.new_publish_to_iot_core()
        operation1.activate(request1)        
        future_response1 = operation1.get_response()
        future_response1.result(TIMEOUT)

        transmit_data_dict2={
            'dataset':'ims_'+ data_set_paths[select_data_set].split(sep='/')[-1],
            'filename':filename,
            'filepart':2,
            'bearing':ch,
            'readings':values[10240:]
            }
        transmit_data_json2=json.dumps(transmit_data_dict2)
        request2 = PublishToIoTCoreRequest()
        request2.topic_name = topic
        payload = transmit_data_json2
        request2.payload = bytes(payload, "utf-8")
        request2.qos = qos
        operation2 = ipc_client.new_publish_to_iot_core()
        operation2.activate(request2)
        future_response2 = operation1.get_response()
        future_response2.result(TIMEOUT)
        

