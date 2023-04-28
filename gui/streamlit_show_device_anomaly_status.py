import boto3
import streamlit as st

import pandas as pd

from io import BytesIO
from pathlib import Path

from time import perf_counter



#To access 's3' without any access key embedded folloWSLg dependencies shall be met:
        # 1. Policy for user : Allow-S3-Passrole-to-EC2, AmazonS3FullAccess
        # 2. Role            : S3Admin
aws_s3 = boto3.resource('s3')
s3_bucket = aws_s3.Bucket('anomaly-detection-from-bearing-vibration-project-bucket')
s3_bucket_objects=[]

for s3_bucket_object in s3_bucket.objects.filter(Prefix='data_ingested/device_status'):
    s3_bucket_objects.append(s3_bucket_object)

s3_device_objects={}
s3_csv_object=''
for s3_object in s3_bucket_objects:
    path_parts = Path(s3_object.key).parts
    if len(path_parts) == 3 and '.csv' in path_parts[2]:
        s3_csv_object=s3_object

print(s3_csv_object)
df = pd.DataFrame()
data = s3_csv_object.get()['Body'].read()
df = pd.read_csv(BytesIO(data), index_col=[0])
print(df)
