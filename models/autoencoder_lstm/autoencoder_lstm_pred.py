# %%
import os
from  sys import path as sys_path

import pandas as pd
import tensorflow as tf

# %%
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

import model_feedinput_pipeline
from model_feedinput_pipeline import CODE_ENV, DATASET_ID
from models.autoencoder_lstm.autoencoder_lstm_main import scale_timefeature_data, prepare_lstm_input, pred_test_autoencoder


# %%
if __name__ == "__main__":
    import sys    

    #####################################################################################
    #***************IMP: Update coding environment********************
    #####################################################################################
    code_env = CODE_ENV.WSL    
    curr_dataset = DATASET_ID.Third


    #Step 1 : Setup Data Source
    dataset_paths = model_feedinput_pipeline.get_dataset_paths(code_env)   

    #Step 2 : Read the time features generated earlier
    time_feature_data_filename=['timefeatures_1st_1.csv', 'timefeatures_1st_2.csv',
                                'timefeatures_2nd.csv', 'timefeatures_3rd.csv']

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

    time_features_data = pd.read_csv(time_feature_data_filename[tf_file_indx])
    print('Number of records in TimeFeatureData=', len(time_features_data))

    restored_model = tf.keras.models.load_model('./bvad_ae_lstm')


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
    X_train, X_test = scale_timefeature_data(train, test)
    X_train, X_test = prepare_lstm_input(X_train, X_test)

    threshold = 1.8828531646427422
    print('Threshold =', threshold)
    # calculate the same metrics for the training set 
    # and merge all data in a single dataframe for plotting
    test_scored1, X_test_pred1, XTest1 = pred_test_autoencoder(restored_model, X_train, train, threshold)
    test_scored2, X_test_pred2, XTest2 = pred_test_autoencoder(restored_model, X_test, test, threshold)
    scored = pd.concat([test_scored1, test_scored2])

    #print(scored)

    # plot bearing failure time plot
    import matplotlib.pyplot as plt
    scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])
    plt.show()

