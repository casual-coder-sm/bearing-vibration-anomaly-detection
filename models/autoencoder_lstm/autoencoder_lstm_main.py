import os
import sys
from sys import path as sys_path
# Code to re-configure current working directory (for ipynb environment)
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

from pathlib import Path

import numpy as np
np.random.seed(10)
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)

from model_feedinput_pipeline import get_dataset_paths
from autoencoder_lstm.Bvad_AutoEncoder import Bvad_AutoEncoder
from autoencoder_lstm.ScalerWrapper import ScalerWrapper
from model_feedinput_pipeline import CODE_ENV, DATASET_ID
from autoencoder_lstm.autoencoder_lstm_generate_features import get_time_features

select_columns = [
    ['b1_ch1', 'b2_ch3', 'b3_ch5', 'b4_ch7']
    , ['b1_ch2', 'b2_ch4', 'b3_ch6', 'b4_ch8']
    , ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
    , ['b1_ch1', 'b2_ch2', 'b3_ch3', 'b4_ch4']
]
time_feature_data_filename = ['ae_timefeatures_1st_1.csv'
                              , 'ae_timefeatures_1st_2.csv'
                              , 'ae_timefeatures_2nd.csv'
                              , 'ae_timefeatures_3rd.csv']
dataset_id_mapping=[
    DATASET_ID.First,
    DATASET_ID.First,
    DATASET_ID.Second,
    DATASET_ID.Third
]
cut_off_date_time =[
    '2003-11-20 23:54:03',
    '2003-11-20 23:54:03',
    '2004-02-15 12:52:39',
    '2004-04-08 23:51:57'
]
scaler_filenames=[
    "ae_scaler_1st_1",
    "ae_scaler_1st_2",
    "ae_scaler_2nd",
    "ae_scaler_3rd"
]

#Test Stage 1 : Setup Data Source
#Call get_dataset_paths with coding environment details

#Test Stage 2 : Generate the timefeatures
def main_generate_timefeatures(code_env:CODE_ENV):
    dataset_paths = get_dataset_paths(code_env)
    for i in range(len(select_columns)):
        time_feature_data = get_time_features(code_env, dataset_paths, dataset_id_mapping[i], select_columns[i])
        time_feature_data.to_csv(time_feature_data_filename[i])
    
    return
    

#Test Stage 3 : Normalize
def main_generate_scalers():
    for tf_file_indx in range(len(scaler_filenames)):
        df_features = pd.read_csv(time_feature_data_filename[tf_file_indx])
        train = df_features[df_features['filename']<=cut_off_date_time[tf_file_indx]]
        test  = df_features[df_features['filename']>cut_off_date_time[tf_file_indx]]
        train = train.set_index('filename')
        test  = test.set_index('filename')
        
        scaler = ScalerWrapper()
        scaler.fit_transform(train)
        scaler.transform(test)
        scaler.save_scaler(scaler_filenames[tf_file_indx])
    
    return


#Test Stage 4 : Train a model based on selected Dataset and validate Dataset
def main_generate_autoencoder_lstm_models(isWithValidation:bool=True, save_plots:bool=False):
    results = {}
    for tf_file_indx in range(0, len(cut_off_date_time)):
        validset_file_indx=(tf_file_indx+2)%4
        print('Validation Set Index=',validset_file_indx)

        df_features = pd.read_csv(time_feature_data_filename[tf_file_indx])
        df_features_valid = pd.read_csv(time_feature_data_filename[validset_file_indx])

        train = df_features[df_features['filename']<=cut_off_date_time[tf_file_indx]]
        test  = df_features[df_features['filename']>cut_off_date_time[tf_file_indx]]
        valid = df_features_valid[df_features_valid['filename']<=cut_off_date_time[validset_file_indx]]

        train = train.set_index('filename')
        test  = test.set_index('filename')
        valid = valid.set_index('filename')
        print(train.shape, test.shape, valid.shape)
        
        scaler = ScalerWrapper()
        scaler.load_scaler(scaler_filenames[tf_file_indx])
        X_scaled_train, X_scaled_test = scaler.transform(X_train=train, X_test=test)

        scaler_valid = ScalerWrapper()
        scaler_valid.load_scaler(scaler_filenames[validset_file_indx])
        _, X_scaled_valid = scaler_valid.transform(X_test=valid)
        X_lstm_valid   = X_scaled_valid.reshape(X_scaled_valid.shape[0], 1, X_scaled_valid.shape[1])

        print(X_scaled_train.shape, X_scaled_test.shape, X_lstm_valid.shape)
        auto_encoder = Bvad_AutoEncoder(train=train, test=test, scaler=scaler)
        if isWithValidation:
            model, history = auto_encoder.train_autoencoder_model(X_valid=X_lstm_valid, nb_epochs=100)
        else:
            model, history = auto_encoder.train_autoencoder_model()
        
        auto_encoder.pred_autoencoder()
        
        #save the model
        model.save('bvad_ae_lstm_'+str(tf_file_indx))

        history, X_train_pred, threshold, train_scored = auto_encoder.get_training_result()
        results[tf_file_indx]={
            'model':model
            , 'history':history
            , 'train_pred': X_train_pred
            , 'threshold': threshold
            , 'score':train_scored
        }
        
        with open('training_results_'+str(tf_file_indx)+'.txt', 'w') as training:
            training.write(str(history))
            training.write(str(threshold))
            training.write(str(train_scored))
        X_train_pred.to_csv('train_pred_output'+str(tf_file_indx)+'.csv')


        # plot the training losses
        if save_plots:
            fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
            ax.plot(history['loss'], 'b', label='Train', linewidth=2)
            ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
            ax.set_title('Model loss', fontsize=16)
            ax.set_ylabel('Loss (mae)')
            ax.set_xlabel('Epoch')
            ax.legend(loc='upper right')
            plt.savefig('Training_Loss_Distribution_'+str(tf_file_indx)+'.png')

        for pred_set_indx in range(len(cut_off_date_time)):
            df_features = pd.read_csv(time_feature_data_filename[pred_set_indx])
            df_features = df_features.set_index('filename')

            restored_model = tf.keras.models.load_model('../bvad_ae_lstm_'+str(tf_file_indx))

            scaler = ScalerWrapper()
            scaler.load_scaler(scaler_filenames[pred_set_indx])
            _, X_scaled_test = scaler.transform(df_features)

            auto_encoder = Bvad_AutoEncoder(test=df_features, scaler=scaler, model=restored_model)
            auto_encoder.pred_autoencoder(threshold=threshold)

            X_test_pred, _, test_scored = auto_encoder.get_test_result()

            # plot bearing failure time plot
            if save_plots:
                test_scored.plot(logy=True,  figsize=(8,4), ylim=[1e-2,1e2], color=['blue','red'])
                plt.savefig('Prediction_Loss_Distribution_'+str(tf_file_indx)+'_'+str(pred_set_indx)+'.png')
        
    return


#Test Stage 5 :  Predict based on previously trained model and scaler fittened on non-anomaly dataset
def main_predict_autoencoder_lstm_models(draw_plots:bool=False):
    recorded_thresholds = [
        2.725298151678263
        , 2.677191266177443
        , 1.1472609170475838
        , 2.2959081453652157
    ]
    results = {}    
    for tf_file_indx in range(len(recorded_thresholds)):
        df_features = pd.read_csv(time_feature_data_filename[tf_file_indx])
        df_features = df_features.set_index('filename')

        scaler = ScalerWrapper()
        scaler.load_scaler(scaler_filenames[tf_file_indx])
        _, X_scaled_test = scaler.transform(df_features)

        restored_model = tf.keras.models.load_model('bvad_ae_lstm_'+str(tf_file_indx))
        auto_encoder = Bvad_AutoEncoder(test=df_features, scaler=scaler, model=restored_model)
        auto_encoder.pred_autoencoder(threshold=recorded_thresholds[tf_file_indx])

        X_test_pred, _, test_scored = auto_encoder.get_test_result()
        results[tf_file_indx]={
            'test_pred':X_test_pred
            , 'threshold':recorded_thresholds[tf_file_indx]
            , 'score':test_scored
        }

        # plot bearing failure time plot
        if draw_plots:
            test_scored.plot(logy=True,  figsize=(8,4), ylim=[1e-2,1e2], color=['blue','red'])

    return results
        

if __name__ == "__main__":
    #####################################################################################
    #***************IMP: Update coding environment********************
    #####################################################################################
    code_env = CODE_ENV.WSL

    #Test Stage 1 : Setup Data Source
    #dataset_paths = get_dataset_paths(code_env)
    #main_generate_timefeatures(code_env)
    #main_generate_scalers()
    #main_generate_autoencoder_lstm_models(True)
    main_predict_autoencoder_lstm_models()
