# %% 
import os
import sys


# %%
import os
from sys import path as sys_path
sys_path.insert(0, '/greengrass/v2/packages/artifacts-unarchived/bearing_vibration_anomaly_detection/1.0.0/models/models')

from model_feedinput_pipeline import CODE_ENV
from model_feedinput_pipeline import DATASET_ID
from model_feedinput_pipeline import get_dataset_paths
from model_feedinput_pipeline import get_df



if __name__ == "__main__":
    select_input_stepsize= 3000

    #####################################################################################
    #***************IMP: Update coding environment********************
    #####################################################################################
    code_env = CODE_ENV.DEV

    #Trial: collect filepath details
    #dataset_details = get_dataset_paths(code_env)

    #Trial: Reading content of file
    #df = get_df(dataset_details, DATASET_ID.First, 0, code_env)    
    #print(df.head())
    

    restored_model = tf.keras.models.load_model('./bvad_ae_lstm')
