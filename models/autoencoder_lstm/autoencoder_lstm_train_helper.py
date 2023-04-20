# %%
import os
from sys import path as sys_path

import pandas as pd
import numpy as np
from numpy.random import seed
seed(10)

import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
sns.set(color_codes=True)

from scipy.stats import entropy

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(10)
#tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers


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


# %% [markdown]
# # Helper Functions

# %%
def scale_timefeature_data(X_train, X_test, saveScaler=False):
    # normalize the data
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    if saveScaler:
        import joblib
        joblib.dump(scaler, open('autoencoder_scaler', 'wb'))
    return scaled_X_train, scaled_X_test


def prepare_lstm_input(X_train, X_test):
    # reshape inputs for LSTM [samples, timesteps, features]
    X_train_out = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_out = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return X_train_out, X_test_out


# define the autoencoder network model
def autoencoder_model(X_train):
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X_train.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X_train.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    
    return model


def train_autoencoder(model, X_train):
    nb_epochs = 100
    batch_size = 10
    history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history
    return model, history


def train_autoencoder_main(time_features_data:pd.DataFrame, cut_off_date_time:str):

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
    X_train, X_test = prepare_lstm_input(X_train, X_test)

    #Step 4 : Build the model
    model = autoencoder_model(X_train)

    #Step 5: Train the model
    history = ''
    model, history = train_autoencoder(model, X_train)

    return train, test, X_train, X_test, model, history


def pred_train_autoencoder(model, X_train, train):
    
    # calculate the loss on the test set
    X_train_pred = model.predict(X_train)
    X_train_pred = X_train_pred.reshape(X_train_pred.shape[0], X_train_pred.shape[2])

    X_train_pred = pd.DataFrame(X_train_pred, columns=train.columns)
    X_train_pred.index = train.index

    train_scored = pd.DataFrame(index=train.index)
    XTrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
    train_scored['Loss_mae'] = np.mean(np.abs(X_train_pred-XTrain), axis = 1)
    threshold = min(train_scored['Loss_mae']) + (max(train_scored['Loss_mae']) - min(train_scored['Loss_mae'])) * 0.8
    print('Chosen Threshold =', threshold)
    train_scored['Threshold'] = threshold
    train_scored['Anomaly'] = train_scored['Loss_mae'] > train_scored['Threshold']
    return train_scored, X_train_pred, XTrain, threshold


def pred_test_autoencoder(model, X_test, test, threshold):
    
    # calculate the loss on the test set
    X_test_pred = model.predict(X_test)
    X_test_pred = X_test_pred.reshape(X_test_pred.shape[0], X_test_pred.shape[2])
    
    X_test_pred = pd.DataFrame(X_test_pred, columns=test.columns)
    X_test_pred.index = test.index

    test_scored = pd.DataFrame(index=test.index)
    XTest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    test_scored['Loss_mae'] = np.mean(np.abs(X_test_pred-XTest), axis = 1)
    test_scored['Threshold'] = threshold
    test_scored['Anomaly'] = test_scored['Loss_mae'] > test_scored['Threshold']
    return test_scored, X_test_pred, XTest


