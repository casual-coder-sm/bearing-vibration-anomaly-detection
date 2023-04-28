import boto3
import streamlit as st

import numpy as np
import pandas as pd

from io import BytesIO
from pathlib import Path

import time
import signal

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        self.kill_now = True

def write_to_s3_bucket(df_st_status):
    df_st_status.to_csv('device_live_status.csv')
    client = boto3.client("s3")
    client.upload_file('device_live_status.csv'
                       , 'anomaly-detection-from-bearing-vibration-project-bucket'
                       , 'data_ingested/device_status/device_live_status.csv')
    

def collect_s3_bucket_object_keys():
    #To access 's3' without any access key embedded folloWSLg dependencies shall be met:
            # 1. Policy for user : Allow-S3-Passrole-to-EC2, AmazonS3FullAccess
            # 2. Role            : S3Admin
    aws_s3 = boto3.resource('s3')
    s3_bucket = aws_s3.Bucket('anomaly-detection-from-bearing-vibration-project-bucket')

    start_time = time.perf_counter()
    s3_bucket_objects=[]
    for s3_bucket_object in s3_bucket.objects.filter(Prefix='data_ingested/bvad/anomaly_status'):
        s3_bucket_objects.append(s3_bucket_object)

    return s3_bucket_objects
    

def get_s3_bucket_object_keys():
    start_time = time.perf_counter()
    s3_bucket_objects = collect_s3_bucket_object_keys()
    s3_device_objects={}
    for s3_object in s3_bucket_objects:
        path_parts = Path(s3_object.key).parts
        if len(path_parts) >= 5:
            if path_parts[3] in s3_device_objects:
                s3_device_objects[path_parts[3]].append(s3_object)
            else:
                s3_device_objects[path_parts[3]] = [s3_object]
    end_time = time.perf_counter()
    print(f"prepared list of files:{end_time-start_time}")
    return s3_device_objects


def get_updated_s3_bucket_object_keys(ref_s3_device_objects, counter):
    start_time = time.perf_counter()
    s3_bucket_objects_updated = collect_s3_bucket_object_keys()
    s3_device_objects_updated={}
    for s3_object in s3_bucket_objects_updated:
        path_parts = Path(s3_object.key).parts
        if len(path_parts) >= 5:
            if s3_object not in ref_s3_device_objects[path_parts[3]]:            
                if path_parts[3] in s3_device_objects_updated:
                    s3_device_objects_updated[path_parts[3]].append(s3_object)
                else:
                    s3_device_objects_updated[path_parts[3]] = [s3_object]
    end_time = time.perf_counter()
    print(f"preparing list for {counter+1} time :{end_time-start_time}")
    return s3_device_objects_updated


def read_from_s3_object(s3_object_keys_list, ref_df):
    df = pd.DataFrame()
    out_df = pd.DataFrame()
    for s3_object in s3_object_keys_list:
        data = s3_object.get()['Body'].read()
        df = pd.read_json(BytesIO(data))
        df = df.reset_index()
        df.drop(columns=['index'], inplace=True)
        #df = df.set_index('filename')
        df = df.sort_values('filename')
        out_df = pd.concat([out_df, df])
    out_df.drop_duplicates(inplace=True, ignore_index=True)
    ref_df = pd.concat([ref_df, out_df[~out_df.index.isin(ref_df.index)]], axis=0)
    return ref_df


if __name__ == "__main__":
    killer=GracefulKiller()

    s3_device_objects = get_s3_bucket_object_keys()
    device_status_list={}
    for key in s3_device_objects:
        device_status_list[key]=pd.DataFrame()

    s3_device_objects_updated = s3_device_objects
    counter=0
    df_st_status = pd.DataFrame()
    df_st_status_updated = pd.DataFrame()
    while not killer.kill_now:
        for key in s3_device_objects_updated:
            start_time = time.perf_counter()
            device_status_list[key] = read_from_s3_object(s3_device_objects_updated[key], device_status_list[key])
            #write output for streamlit dataframe
            df_in = device_status_list[key]
            df_out=pd.DataFrame(df_in[-1:],columns=df_in.columns)
            df_st_status_updated = pd.concat([df_st_status_updated, df_out], axis=0)
            end_time = time.perf_counter()
            print(f"Execution time for processing {key} : {end_time-start_time}")
            print(device_status_list[key].shape, device_status_list[key].columns)

        df_st_status_updated = df_st_status_updated.reset_index().drop(columns=['index'])
        if counter==0:
            df_st_status = df_st_status_updated
        else:
            df_cmp = df_st_status.compare(df_st_status_updated, align_axis=0)
            if df_cmp.shape[0] >0:
                df_st_status=df_st_status_updated
                write_to_s3_bucket(df_st_status)
            else:
                print('No update skip iteration')        
        
        start_time = time.perf_counter()
        counter = counter+1
        s3_device_objects_updated = get_updated_s3_bucket_object_keys(s3_device_objects, counter)        
        for key in s3_device_objects:
            if key not in s3_device_objects_updated:
                print('No update for ', key)
                                
        for key in s3_device_objects_updated:
            print('before :', s3_device_objects[key].shape)
            print('updates:', s3_device_objects_updated[key].shape)
            for s3_object in s3_device_objects_updated[key]:
                s3_device_objects[key].append(s3_object)
            print('after  :', s3_device_objects[key].shape)
        print(f"Execution time for updating s3 device object list : {end_time-start_time}")

        time.sleep(5)
        
    print('End of the program. Exited Gracefully')
        
        

