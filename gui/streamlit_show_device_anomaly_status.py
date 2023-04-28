import boto3
import streamlit as st

import pandas as pd

from io import BytesIO
from pathlib import Path

from time import perf_counter


def color_status(val):
    color = '#AAFF00' if val == 'Operational' else '#FF5733'
    return f'background-color: {color}'


def status(val):
    text = 'Operational' if val else 'Error'
    color = '#AAFF00' if val else '#FF5733'
    return f'<p style="background:{color}; width:30px; height:30px; border-radius:50%;text-indent: -9999px;">{text}</p>'


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

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=5)
def get_device_live_status(s3_csv_object):
    df = pd.DataFrame()
    data = s3_csv_object.get()['Body'].read()
    df = pd.read_csv(BytesIO(data), index_col=[0])
    df['Anomaly']=df['Anomaly'].apply(status)
    df = df.to_html(escape=False)

df = get_device_live_status(s3_csv_object)
st.write(df, unsafe_allow_html=True)


