from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class ScalerWrapper(StandardScaler):
    def __init__(self):
        from enum import Enum
        class SCALER_TYPE(Enum):
            SkLearn = 0
            JobLib  = 1        
        self.scaler_type = SCALER_TYPE.SkLearn
        self.is_transformed = False
        pass

    def fit_transform(self, X):
        if self.scaler_type == ScalerWrapper.SCALER_TYPE.SkLearn:
            scaled_X=self.scaler.fit_transform(X)            
            self.is_transformed = True
        else:
            raise TypeError('Call transform for Joblib based Scaler')        
        
        return scaled_X
    
    def transform(self, X_test):
        if self.is_transformed:
            scaled_X_test=self.scaler.transform(X_test)
        else:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(self, "Call fit_transform")        
        
        return scaled_X_test
    

    def save_scaler(self, scaler_file_output_path):
        import joblib
        joblib.dump(self.scaler, open(scaler_file_output_path, 'wb'))


    def load_scaler(self, scaler_file_input_path):
        import joblib
        self.scaler = joblib.load(open(scaler_file_input_path,'rb'))
        self.is_transformed = True

        