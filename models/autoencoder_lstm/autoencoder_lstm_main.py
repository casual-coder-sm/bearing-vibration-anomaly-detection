# %%
import os
from sys import path as sys_path
from pathlib import Path

import numpy as np
np.random.seed(10)
import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
sns.set(color_codes=True)


# %%
import sys
from sys import path as sys_path

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


# %%
def train_autoencoder_wrapper(time_features_data:pd.DataFrame, cut_off_date_time:str):
    #Step 1 : Re-construct the index columns 'date_time'    
    #print(time_features_data.describe().T)
    #print(time_features_data.columns)
    time_features_data=time_features_data.rename(columns={'filename':'date_time'})
    time_features_data['date_time']=pd.to_datetime(time_features_data['date_time'])   
    
    #Step 2 : Prepare Train, Validation and Test test
    train = time_features_data[time_features_data['date_time'] <= cut_off_date_time]
    test  = time_features_data[time_features_data['date_time'] > cut_off_date_time]
    train = train.set_index('date_time')
    test  = test.set_index('date_time')

    #Step 3 : Prepare Data : Normalize & Reshape
    X_train, X_test = scale_timefeature_data(train, test, True)
    return 


# %%
from autoencoder_lstm import Bvad_AutoEncoder
from autoencoder_lstm import ScalerWrapper
from model_feedinput_pipeline import CODE_ENV, DATASET_ID
from autoencoder_lstm.autoencoder_lstm_generate_features import get_time_features


# %%
# %%
def get_dataset_paths(code_env:CODE_ENV)->dict:

    #Step1 : set the pre-defined root folder paths
    if code_env == CODE_ENV.EC2:
        #To access 's3' without any access key embedded folloWSLg dependencies shall be met:
        # 1. Policy for user : Allow-S3-Passrole-to-EC2, AmazonS3FullAccess
        # 2. Role            : S3Admin
        import boto3
        aws_s3 = boto3.resource('s3')
        s3_bucket = aws_s3.Bucket('anomaly-detection-from-bearing-vibration-project-bucket')
        s3_bucket_objects=[]
        for s3_bucket_object in s3_bucket.objects.all():
            s3_bucket_objects.append(s3_bucket_object)
    elif code_env == CODE_ENV.WSL:
        ##################################################################################
        #SET PATH to DATASET in your PC (till but not including folder named '1st')
        #CUSTOMIZE AS per your PC setup
        #curr_dir=os.getcwd()        
        curr_dir = str(Path('/mnt/g/My Drive/CDS/github/CDS/capstone_project/'))
        dataset_root_path =Path('')
        dataset_root_path1 = Path(curr_dir+'/'+'capstone-data/01_PHM-Bearing')
        dataset_root_path2 = Path(curr_dir+'/'+'models/capstone-data/01_PHM-Bearing')
        if dataset_root_path1.is_dir():
            dataset_root_path = dataset_root_path1
        elif dataset_root_path2.is_dir():
            dataset_root_path = dataset_root_path2
        else:
            print('Path ERROR!!!', str(dataset_root_path1), str(dataset_root_path2))
        ##################################################################################
    elif code_env == CODE_ENV.DEV:
        dataset_root_path = Path('/root/datasets/phm-ims-datasets')
        if not dataset_root_path.is_dir():
            print('Check Folder path = ', dataset_root_path.as_posix())
        
    
    #Step2 : collect 3 dataset file details
    if code_env == CODE_ENV.EC2:
        s3_objects_1st_dataset=[]
        s3_objects_2nd_dataset=[]
        s3_objects_3rd_dataset=[]
        paths = []
        for s3_object in s3_bucket_objects:
            path_parts = Path(s3_object.key).parts
            if len(path_parts) == 4 and path_parts[0] == 'data_input' and path_parts[1] == 'IMS':
                paths.append(s3_object)
                if path_parts[2] == '1st_test':
                    s3_objects_1st_dataset.append(s3_object)
                elif path_parts[2] == '2nd_test':
                    s3_objects_2nd_dataset.append(s3_object)
                else:
                    s3_objects_3rd_dataset.append(s3_object)
    elif code_env == CODE_ENV.WSL or code_env == CODE_ENV.DEV:
        data_set1_path = dataset_root_path.as_posix() + '/1st_test'
        data_set2_path = dataset_root_path.as_posix() + '/2nd_test'
        data_set3_path = dataset_root_path.as_posix() + '/3rd_test'
        filelist_1st_dataset = [data_set1_path+'/'+src_path for src_path in sorted(os.listdir(data_set1_path))]
        filelist_2nd_dataset = [data_set2_path+'/'+src_path for src_path in sorted(os.listdir(data_set2_path))]
        filelist_3rd_dataset = [data_set3_path+'/'+src_path for src_path in sorted(os.listdir(data_set3_path))]
    
    #Step3 : Consolidate to structure => Output
    #1st Set has 8 
    col_names_1st = ['b1_ch1', 'b1_ch2', 'b2_ch3', 'b2_ch4', 'b3_ch5', 'b3_ch6', 'b4_ch7', 'b4_ch8']
    #2nd and 3rd has 4
    col_names_2nd_3rd = ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']

    dataset_details = {}
    dataset_details[DATASET_ID.First]  = {'col_names':col_names_1st}
    dataset_details[DATASET_ID.Second] = {'col_names':col_names_2nd_3rd}
    dataset_details[DATASET_ID.Third]  = {'col_names':col_names_2nd_3rd}

    if code_env == CODE_ENV.EC2:
        dataset_details[DATASET_ID.First]['paths']=s3_objects_1st_dataset
        dataset_details[DATASET_ID.Second]['paths']=s3_objects_2nd_dataset
        dataset_details[DATASET_ID.Third]['paths']=s3_objects_3rd_dataset
        #Verify variables
        #print('Number of files in 1st Dataset:', len(dataset_details[DATASET_ID.First]['paths']), 'first file=', dataset_details[DATASET_ID.First]['paths'][0].key)
        #print('Number of files in 2nd Dataset:', len(dataset_details[DATASET_ID.Second]['paths']), 'first file=', dataset_details[DATASET_ID.Second]['paths'][0].key)
        #print('Number of files in 3rd Dataset:', len(dataset_details[DATASET_ID.Third]['paths']), 'first file=', dataset_details[DATASET_ID.Third]['paths'][0].key)

    elif code_env == CODE_ENV.WSL or code_env == CODE_ENV.DEV:
        dataset_details[DATASET_ID.First]['paths']=filelist_1st_dataset
        dataset_details[DATASET_ID.Second]['paths']=filelist_2nd_dataset
        dataset_details[DATASET_ID.Third]['paths']=filelist_3rd_dataset
        #Verify variables
        #print('Number of files in 1st Dataset:', len(dataset_details[DATASET_ID.First]['paths']), 'first file=', dataset_details[DATASET_ID.First]['paths'][0])
        #print('Number of files in 2nd Dataset:', len(dataset_details[DATASET_ID.Second]['paths']), 'first file=', dataset_details[DATASET_ID.Second]['paths'][0])
        #print('Number of files in 3rd Dataset:', len(dataset_details[DATASET_ID.Third]['paths']), 'first file=', dataset_details[DATASET_ID.Third]['paths'][0])
    
    return dataset_details



# %%
if __name__ == "__main__":
    #####################################################################################
    #***************IMP: Update coding environment********************
    #####################################################################################
    code_env = CODE_ENV.WSL

    #Step 1 : Setup Data Source
    dataset_paths = get_dataset_paths(code_env)   

    #Step 2 : Generate the timefeatures
    select_columns = [
        ['b1_ch1', 'b2_ch3', 'b3_ch5', 'b4_ch7']
      , ['b1_ch2', 'b2_ch4', 'b3_ch6', 'b4_ch8']
      , ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
      , ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
    ]
    time_feature_data_filename = ['timefeatures_1st_1.csv'
                                  , 'timefeatures_1st_2.csv'
                                  , 'timefeatures_2nd.csv'
                                  , 'timefeatures_3rd.csv']

    time_feature_data = []
    for i in range(len(select_columns)):
        time_feature_data.append(time_feature_data.append(get_time_features(code_env, dataset_paths, DATASET_ID(i), select_columns[i])))

    for i in range(len(select_columns)):
        time_feature_data[i].to_csv(time_feature_data_filename[i])

    tf_file_indx = 0
    cut_off_date_time =[
        '2003-11-20 23:54:03',
        '2003-11-20 23:54:03',
        '2004-02-15 12:52:39',
        '2004-04-08 23:51:57'
    ]

        