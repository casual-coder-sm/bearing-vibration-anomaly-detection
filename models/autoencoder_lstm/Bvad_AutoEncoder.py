import numpy as np
import pandas as pd

import tensorflow as tf
tf.random.set_seed(10)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

from ScalerWrapper import ScalerWrapper

class Bvad_AutoEncoder:
    '''
        Perform AutoEndoer LSTM on normalized (Scaled) dataset.
        Supply both train and test for training.
        Supply Only test for Prediction.
        Supply Validation set sepearately while calling fit method
    '''
    def __init__(self, train:pd.DataFrame=None, test:pd.DataFrame=None, scaler:ScalerWrapper=None, model=None) -> None:
        
        self.train = train
        self.test  = test

        self.scaled_X_train, self.scaled_X_test = scaler.get_scaled()

        #Type: numpy array
        self.X_lstm_train = None
        self.X_lstm_test = None

        #Type numpy array
        self.X_train_pred = None
        self.X_test_pred = None

        self.scaler=scaler
        self.model = model
        
        self.history = None
        self.threshold = None

        self.train_scored = None
        self.test_scored = None
        pass


    def prepare_lstm_input(self):
        # reshape inputs for LSTM [samples, timesteps, features]
        if self.scaled_X_train is None and self.scaled_X_test is None:
            import sklearn.exceptions as sklearn_exception
            raise sklearn_exception.DataDimensionalityWarning
        
        if self.scaled_X_train is not None:
            self.X_lstm_train = self.scaled_X_train.reshape(self.scaled_X_train.shape[0], 1, self.scaled_X_train.shape[1])
        
        if self.scaled_X_test is not None:
            self.X_lstm_test = self.scaled_X_test.reshape(self.scaled_X_test.shape[0], 1, self.scaled_X_test.shape[1])
        
        return        


    # define the autoencoder network model
    def train_autoencoder_model(self, X_valid=None, nb_epochs=100, batch_size=10):
        import sklearn.exceptions as sklearn_exception
        
        self.prepare_lstm_input()

        if self.X_lstm_train is None:
            import sklearn.exceptions as sklearn_exception
            raise sklearn_exception.DataDimensionalityWarning

        #Create Model
        inputs = Input(shape=(self.X_lstm_train.shape[1], self.X_lstm_train.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(self.X_lstm_train.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(self.X_lstm_train.shape[2]))(L5)    
        self.model = Model(inputs=inputs, outputs=output)

        #Compile
        self.model.compile(optimizer='adam', loss='mae')

        #Train
        self.history = self.model.fit(self.X_lstm_train, self.X_lstm_train, batch_size=batch_size, epochs=nb_epochs
                                 ,verbose=1, validation_data=X_valid, validation_split=0.5)
        return self.model, self.history.history
    

    def pred_autoencoder(self, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        
        self.prepare_lstm_input()

        if self.X_lstm_train is not None:
            if self.train is None:
                import sklearn.exceptions as sklearn_exception
                raise sklearn_exception.DataDimensionalityWarning
        
            # calculate the loss on the test set
            self.X_train_pred = self.model.predict(self.X_lstm_train)
            self.X_train_pred = self.X_train_pred.reshape(self.X_train_pred.shape[0], self.X_train_pred.shape[2])
            self.X_train_pred = pd.DataFrame(self.X_train_pred, columns=self.train.columns)
            self.X_train_pred.index = self.train.index

            self.train_scored = pd.DataFrame(index=self.train.index)
            self.train_scored['Loss_mae'] = np.mean(np.abs(self.X_train_pred-self.scaled_X_train), axis = 1)
            self.threshold = min(self.train_scored['Loss_mae']) + (max(self.train_scored['Loss_mae']) - min(self.train_scored['Loss_mae'])) * 0.8
            print('Chosen Threshold =', self.threshold)
            self.train_scored['Threshold'] = self.threshold
            self.train_scored['Anomaly'] = self.train_scored['Loss_mae'] > self.train_scored['Threshold']
        elif self.threshold is None:
            import sklearn.exceptions as sklearn_exception
            raise sklearn_exception.NotFittedError

        if self.X_lstm_test is not None:
            if self.test is None:
                import sklearn.exceptions as sklearn_exception
                raise sklearn_exception.DataDimensionalityWarning
            
            # calculate the loss on the test set
            self.X_test_pred = self.model.predict(self.X_lstm_test)
            self.X_test_pred = self.X_test_pred.reshape(self.X_test_pred.shape[0], self.X_test_pred.shape[2])
            self.X_test_pred = pd.DataFrame(self.X_test_pred, columns=self.test.columns)
            self.X_test_pred.index = self.test.index
            
            self.test_scored = pd.DataFrame(index=self.test.index)
            self.test_scored['Loss_mae'] = np.mean(np.abs(self.X_test_pred-self.scaled_X_test), axis = 1)
            self.test_scored['Threshold'] = self.threshold
            self.test_scored['Anomaly'] = self.test_scored['Loss_mae'] > self.test_scored['Threshold']

        return

            
    def get_training_history(self):
        return self.history
    

    def get_training_model(self):
        return self.model
    

    def get_training_result(self):
        return self.history.history, self.X_train_pred, self.threshold, self.train_scored
    

    def get_test_result(self):
        return self.X_test_pred, self.threshold, self.test_scored
    
