from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

class SCALER_TYPE(Enum):
    SkLearn = 0
    JobLib  = 1 

class ScalerWrapper():
    def __init__(self):
        self.scaler_type = SCALER_TYPE.SkLearn
        self.is_transformed = False
        self.scaler = StandardScaler()
        self.scaled_X_train = None
        self.scaled_X_test  = None
        pass

    def fit_transform(self, X_train):
        if self.scaler_type == SCALER_TYPE.SkLearn:
            self.scaled_X_train=self.scaler.fit_transform(X_train)
            self.is_transformed = True
        else:
            raise TypeError('Call transform for Joblib based Scaler')        
        
        return self.scaled_X_train
    
    def transform(self, X_test, X_train=None):
        if self.is_transformed:
            self.scaled_X_test=self.scaler.transform(X_test)
            if X_train is not None:
                self.scaled_X_train=self.scaler.transform(X_train)
        else:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(self, "Call fit_transform")        
        
        return self.scaled_X_train, self.scaled_X_test
    

    def save_scaler(self, scaler_file_output_path):
        import joblib
        joblib.dump(self.scaler, open(scaler_file_output_path, 'wb'))


    def load_scaler(self, scaler_file_input_path):
        import joblib
        self.scaler = joblib.load(open(scaler_file_input_path,'rb'))
        self.is_transformed = True
    

    def get_scaled(self):
        return self.scaled_X_train, self.scaled_X_test

        