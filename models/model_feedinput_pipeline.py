# %%
import os
from pathlib import Path
import pandas as pd

from io import BytesIO


# %%
from enum import Enum
class CODE_ENV(Enum):
    EC2=0 #Running in AWS EC2
    DEV=1 #Running in IOT Device
    WSL=2 #Running in PC Wsl Environment
#print(list(CODE_ENV))

class DATASET_ID(Enum):
    First=0
    Second=1
    Third=2


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
def get_df(dataset_details:dict, dataset:DATASET_ID, file_index:int, code_env:CODE_ENV):
    df = pd.DataFrame()
    if code_env == CODE_ENV.EC2:
        s3_object = dataset_details[dataset]['paths'][file_index]
        data = s3_object.get()['Body'].read()
        df = pd.read_csv(BytesIO(data), header=None, delimiter='\t', names=dataset_details[dataset]['col_names'], low_memory='False')
    elif code_env == CODE_ENV.WSL or code_env == CODE_ENV.DEV:
        df = pd.read_csv(dataset_details[dataset]['paths'][file_index], header=None, delimiter='\t', names=dataset_details[dataset]['col_names'], low_memory='False')
    
    return df


# %%
if __name__ == "__main__":
    select_input_stepsize= 3000

    #####################################################################################
    #***************IMP: Update coding environment********************
    #####################################################################################
    code_env = CODE_ENV.WSL

    #Trial: collect filepath details
    dataset_details = get_dataset_paths(code_env)

    #Trial: Reading content of file
    df = get_df(dataset_details, DATASET_ID.First, 0, code_env)
    print(df.head())


# %%



